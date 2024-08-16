# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import math
import re
import time
import random
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import apex
from .optim import get_optimizer
from .utils import to_cuda, concat_batches, find_modules
from .utils import parse_lambda_config, update_lambdas
from .file_io import *
from .model.memory import HashingMemory
from .model.transformer import TransformerFFN, PredLayer
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from collections import defaultdict
import sacrebleu

logger = getLogger()


class Trainer(object):

    def __init__(self, data, params):
        """
        Initialize trainer.
        """
        # epoch / iteration size
        self.epoch_size = params.epoch_size
        self.writer = SummaryWriter(params.dump_path + '/' + params.exp_name + '_log')

        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # data iterators
        self.iterators = {}

        # list memory components
        self.memory_list = []
        self.ffn_list = []
        for name in self.MODEL_NAMES:
            find_modules(getattr(self, name), f'self.{name}', HashingMemory, self.memory_list)
            find_modules(getattr(self, name), f'self.{name}', TransformerFFN, self.ffn_list)
        logger.info("Found %i memories." % len(self.memory_list))
        logger.info("Found %i FFN." % len(self.ffn_list))

        # set parameters
        self.set_parameters()

        # float16 / distributed (no AMP)
        assert params.amp >= 1 or not params.fp16
        assert params.amp >= 0 or params.accumulate_gradients == 1
        if params.multi_gpu and params.amp == -1:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            for name in self.MODEL_NAMES:
                setattr(self, name, nn.parallel.DistributedDataParallel(getattr(self, name),
                                                                        device_ids=[params.local_rank],
                                                                        output_device=params.local_rank,
                                                                        broadcast_buffers=True))

        logger.info('Using nn.parallel.DistributedDataParallel ...')

        # set optimizers
        self.set_optimizers()

        # float16 / distributed (AMP)
        if params.amp >= 0:
            self.init_amp()
            if params.multi_gpu:
                logger.info("Using apex.parallel.DistributedDataParallel ...")
                for name in self.MODEL_NAMES:
                    setattr(self, name,
                            apex.parallel.DistributedDataParallel(getattr(self, name), delay_allreduce=True))

        # stopping criterion used for early stopping
        if params.stopping_criterion != '':
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == '_':
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # probability of masking out / randomize / not modify words to predict
        params.word_pred_probs = torch.FloatTensor(
            [params.word_mask, params.word_keep, params.word_rand])

        # same for regions
        # keep <do nothing> as choice as well to simplify image masking ops
        params.region_pred_probs = torch.FloatTensor([0.0, params.region_mask, params.region_keep, params.region_rand])
        # multiply by the conditional prob
        params.region_pred_probs.mul_(params.region_pred)
        params.region_pred_probs[0] = 1 - params.region_pred_probs[1:].sum()

        # probabilty to predict a word
        # '<s>': 0, '</s>': 0, '<pad>': 0, '<unk>': 0, '<special0>': 0, '<special1>': 0,
        # '<special2>': 0, '<special3>': 0, '<special4>': 0, '<special5>': 0, '<special6>': 0, '<special7>': 0,
        # '<special8>': 0, '<special9>': 0,
        counts = np.array(list(self.data['dico'].counts.values()))
        params.mask_scores = np.maximum(counts, 1) ** -params.sample_alpha
        params.mask_scores[params.pad_index] = 0  # do not predict <pad> index
        params.mask_scores[counts == 0] = 0  # do not predict <special0~9> tokens
        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sentences = 0
        if 'para' in data or 'vpara' in data:
            d = data.get('para', data.get('vpara'))
            self.stats = OrderedDict(
                [('processed_s', 0), ('processed_w', 0)] +
                [('CLM-%s' % ll, []) for ll in params.langs] +
                [('CLM-%s-%s' % (l1, l2), []) for l1, l2 in d.keys()] +
                [('CLM-%s-%s' % (l2, l1), []) for l1, l2 in d.keys()] +
                [('MLM-%s' % ll, []) for ll in params.langs] +
                [('MLM-%s-%s' % (l1, l2), []) for l1, l2 in d.keys()] +
                [('MLM-%s-%s' % (l2, l1), []) for l1, l2 in d.keys()] +
                [('VLM-%s' % ll, []) for ll in params.langs] +
                [('VLM-%s-tloss' % ll, []) for ll in params.langs] +
                [('VLM-%s-tacc' % ll, []) for ll in params.langs] +
                [('VLM-%s-vloss' % ll, []) for ll in params.langs] +
                [('VLM-%s-vacc' % ll, []) for ll in params.langs] +
                [('VLM-%s-%s' % (l1, l2), []) for l1, l2 in d.keys()] +
                [('VLM-%s-%s-tloss' % (l1, l2), []) for l1, l2 in d.keys()] +
                [('VLM-%s-%s-tacc' % (l1, l2), []) for l1, l2 in d.keys()] +
                [('VLM-%s-%s-vloss' % (l1, l2), []) for l1, l2 in d.keys()] +
                [('VLM-%s-%s-vacc' % (l1, l2), []) for l1, l2 in d.keys()] +
                [('VLM-%s-%s' % (l2, l1), []) for l1, l2 in d.keys()] +
                [('VLM-%s-%s-tloss' % (l2, l1), []) for l1, l2 in d.keys()] +
                [('VLM-%s-%s-tacc' % (l2, l1), []) for l1, l2 in d.keys()] +
                [('VLM-%s-%s-vloss' % (l2, l1), []) for l1, l2 in d.keys()] +
                [('VLM-%s-%s-vacc' % (l2, l1), []) for l1, l2 in d.keys()] +
                [('VMLM-%s' % ll, []) for ll in params.langs] +
                [('VMLM-%s-tloss' % ll, []) for ll in params.langs] +
                [('VMLM-%s-tacc' % ll, []) for ll in params.langs] +
                [('VMLM-%s-vloss' % ll, []) for ll in params.langs] +
                [('VMLM-%s-vacc' % ll, []) for ll in params.langs] +
                [('PC-%s-%s' % (l1, l2), []) for l1, l2 in params.pc_steps] +
                [('AE-%s' % lang, []) for lang in params.ae_steps] +
                [('VAE-%s' % lang, []) for lang in params.vae_steps] +
                [('MT-%s-%s' % (l1, l2), []) for l1, l2 in params.mt_steps] +
                [('RL-%s-%s' % (l1, l2), []) for l1, l2 in params.mmt_steps] +
                [('MMT-%s-%s' % (l1, l2), []) for l1, l2 in params.mmt_steps] +
                [('BT-%s-%s-%s' % (l1, l2, l3), []) for l1, l2, l3 in params.bt_steps] +
                [('VBT-%s-%s-%s' % (l1, l2, l3), []) for l1, l2, l3 in params.vbt_steps] +
                [('IC-%s-%s' % ("img", l1), []) for l1, _, _ in params.vbt_steps] +
                [('MASS-%s' % lang, []) for lang in params.mass_steps] +
                [('VMASS-%s' % lang, []) for lang in params.vmass_steps]
            )
        else:
            self.stats = OrderedDict(
                [('processed_s', 0), ('processed_w', 0)] +
                [('CLM-%s' % ll, []) for ll in params.langs] +
                [('MLM-%s' % ll, []) for ll in params.langs] +
                [('MLM-%s-tloss' % ll, []) for ll in params.langs] +
                [('MLM-%s-tacc' % ll, []) for ll in params.langs] +
                [('VLM-%s' % ll, []) for ll in params.langs] +
                [('VLM-%s-tloss' % ll, []) for ll in params.langs] +
                [('VLM-%s-tacc' % ll, []) for ll in params.langs] +
                [('VLM-%s-vloss' % ll, []) for ll in params.langs] +
                [('VLM-%s-vacc' % ll, []) for ll in params.langs] +
                [('VMLM-%s' % ll, []) for ll in params.langs] +
                [('VMLM-%s-tloss' % ll, []) for ll in params.langs] +
                [('VMLM-%s-tacc' % ll, []) for ll in params.langs] +
                [('VMLM-%s-vloss' % ll, []) for ll in params.langs] +
                [('VMLM-%s-vacc' % ll, []) for ll in params.langs] +
                [('PC-%s-%s' % (l1, l2), []) for l1, l2 in params.pc_steps] +
                [('AE-%s' % lang, []) for lang in params.ae_steps] +
                [('VAE-%s' % lang, []) for lang in params.vae_steps] +
                [('MT-%s-%s' % (l1, l2), []) for l1, l2 in params.mt_steps] +
                [('RL-%s-%s' % (l1, l2), []) for l1, l2 in params.mmt_steps] +
                [('MMT-%s-%s' % (l1, l2), []) for l1, l2 in params.mmt_steps] +
                [('BT-%s-%s-%s' % (l1, l2, l3), []) for l1, l2, l3 in params.bt_steps] +
                [('VBT-%s-%s-%s' % (l1, l2, l3), []) for l1, l2, l3 in params.vbt_steps] +
                [('IC-%s-%s' % ("img", l1), []) for l1, _, _ in params.vbt_steps] +
                [('MASS-%s' % lang, []) for lang in params.mass_steps] +
                [('VMASS-%s' % lang, []) for lang in params.vmass_steps]
            )

        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # initialize lambda coefficients and their configurations
        parse_lambda_config(params)

    def set_parameters(self):
        """
        Set parameters.
        """
        params = self.params
        self.parameters = {}
        named_params = []
        for name in self.MODEL_NAMES:
            named_params.extend([(k, p) for k, p in getattr(self, name).named_parameters() if p.requires_grad])

        # model (excluding memory values)
        self.parameters['model'] = [p for k, p in named_params if not k.endswith(HashingMemory.MEM_VALUES_PARAMS)]

        # memory values
        if params.use_memory:
            self.parameters['memory'] = [p for k, p in named_params if k.endswith(HashingMemory.MEM_VALUES_PARAMS)]
            assert len(self.parameters['memory']) == len(params.mem_enc_positions) + len(params.mem_dec_positions)

        # log
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizers(self):
        """
        Set optimizers.
        """
        params = self.params
        self.optimizers = {}

        # model optimizer (excluding memory values)
        self.optimizers['model'] = get_optimizer(self.parameters['model'], params.optimizer)

        # memory values optimizer
        if params.use_memory:
            self.optimizers['memory'] = get_optimizer(self.parameters['memory'], params.mem_values_optimizer)

        # log
        logger.info("Optimizers: %s" % ", ".join(self.optimizers.keys()))

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        params = self.params
        assert params.amp == 0 and params.fp16 is False or params.amp in [1, 2, 3] and params.fp16 is True
        opt_names = self.optimizers.keys()
        models = [getattr(self, name) for name in self.MODEL_NAMES]
        models, optimizers = apex.amp.initialize(models, [self.optimizers[k] for k in opt_names],
                                                 opt_level=('O%i' % params.amp))
        for name, model in zip(self.MODEL_NAMES, models):
            setattr(self, name, model)
        self.optimizers = {opt_name: optimizer for opt_name, optimizer in zip(opt_names, optimizers)}

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            # exit()

        params = self.params

        # optimizers
        names = self.optimizers.keys()
        optimizers = [self.optimizers[k] for k in names]
        # regular optimization
        if params.amp == -1:
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                for name in names:
                    # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in self.parameters[name]])) ** 0.5
                    clip_grad_norm_(self.parameters[name], params.clip_grad_norm)
                    # norm_check_b = (sum([p.grad.norm(p=2).item() ** 2 for p in self.parameters[name]])) ** 0.5
                    # print(name, norm_check_a, norm_check_b)
            for optimizer in optimizers:
                optimizer.step()

        # AMP optimization
        else:
            if self.n_iter % params.accumulate_gradients == 0:
                with apex.amp.scale_loss(loss, optimizers) as scaled_loss:
                    scaled_loss.backward()
                if params.clip_grad_norm > 0:
                    for name in names:
                        # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in apex.amp.master_params(self.optimizers[name])])) ** 0.5
                        clip_grad_norm_(apex.amp.master_params(self.optimizers[name]), params.clip_grad_norm)
                        # norm_check_b = (sum([p.grad.norm(p=2).item() ** 2 for p in apex.amp.master_params(self.optimizers[name])])) ** 0.5
                        # print(name, norm_check_a, norm_check_b)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                with apex.amp.scale_loss(loss, optimizers, delay_unscale=True) as scaled_loss:
                    scaled_loss.backward()

    def mass_optimize(self, loss, modules):
        """
        Optimize.
        """
        if type(modules) is str:
            modules = [modules]

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # zero grad
        for module in modules:
            self.optimizers[module].zero_grad()

        # backward
        if self.params.fp16:
            assert len(modules) == 1, "fp16 not implemented for more than one module"
            self.optimizers[module].backward(loss)
        else:
            loss.backward()

        # clip gradients
        if self.params.clip_grad_norm > 0:
            for module in modules:
                if self.params.fp16:
                    self.optimizers[module].clip_master_grads(self.params.clip_grad_norm)
                else:
                    clip_grad_norm_(getattr(self, module).parameters(), self.params.clip_grad_norm)

        # optimization step
        for module in modules:
            self.optimizers[module].step()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        update_lambdas(self.params, self.n_total_iter)
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % 5 != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = ' || '.join(['{}: {:7.4f}'.format(k, np.mean(v)) for k, v in self.stats.items()
                              if type(v) is list and len(v) > 0])

        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        s_lr = " - "
        for k, v in self.optimizers.items():
            s_lr = s_lr + (" - %s LR: " % k) + " / ".join("{:.4e}".format(group['lr']) for group in v.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(self.stats['processed_s'] * 1.0 / diff,
                                                               self.stats['processed_w'] * 1.0 / diff)
        self.stats['processed_s'] = 0
        self.stats['processed_w'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def get_iterator(self, iter_name, lang1, lang2, stream):
        """
        Create a new iterator for a dataset.
        """
        logger.info("Creating new training data iterator (%s) ..." % ','.join(
            [str(x) for x in [iter_name, lang1, lang2] if x is not None]))
        assert stream or not self.params.use_memory or not self.params.mem_query_batchnorm
        if lang2 is None:
            if stream:
                iterator = self.data['mono_stream'][lang1]['train'].get_iterator(shuffle=True)
            else:

                iterator = self.data['mono'][lang1]['train'].get_iterator(shuffle=True,
                                                                          group_by_size=self.params.group_by_size,
                                                                          n_sentences=-1)
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            key = 'vpara' if iter_name == 'mmt' else "para"
            iterator = self.data[key][(_lang1, _lang2)]['train'].get_iterator(
                shuffle=True, group_by_size=self.params.group_by_size, n_sentences=-1)

        self.iterators[(iter_name, lang1, lang2)] = iterator
        return iterator

    def get_iterator_vpara(self, iter_name, lang1, lang2, stream):
        """
        Create a new iterator for a dataset.
        """
        logger.info("Creating new training data iterator (%s) ..." % ','.join(
            [str(x) for x in [iter_name, lang1, lang2] if x is not None]))
        assert stream or not self.params.use_memory or not self.params.mem_query_batchnorm
        if lang2 is None:
            if stream:
                iterator = self.data['vmono_stream'][lang1]['train'].get_iterator(shuffle=True)
            else:
                iterator = self.data['vmono'][lang1]['train'].get_iterator(shuffle=True,
                                                                           group_by_size=self.params.group_by_size,
                                                                           n_sentences=-1)
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            iterator = self.data['vpara'][(_lang1, _lang2)]['train'].get_iterator(
                shuffle=True, group_by_size=self.params.group_by_size, n_sentences=-1)

        self.iterators[(iter_name, lang1, lang2)] = iterator
        return iterator

    def get_batch(self, iter_name, lang1, lang2=None, stream=False):
        """
        Return a batch of sentences from a dataset.
        """
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None
        iterator = self.iterators.get((iter_name, lang1, lang2), None)
        if iterator is None:
            iterator = self.get_iterator(iter_name, lang1, lang2, stream)
        try:
            x = next(iterator)
        except StopIteration:
            iterator = self.get_iterator(iter_name, lang1, lang2, stream)
            x = next(iterator)
        return x if lang2 is None or lang1 < lang2 else x[::-1]

    def get_batch_vpara(self, iter_name, lang1, lang2=None, stream=False):
        """
        Return a batch of sentences and image features from a dataset.
        """
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None
        iterator = self.iterators.get((iter_name, lang1, lang2), None)
        if iterator is None:
            iterator = self.get_iterator_vpara(iter_name, lang1, lang2, stream)
        try:
            x = next(iterator)
        except StopIteration:
            iterator = self.get_iterator_vpara(iter_name, lang1, lang2, stream)
            x = next(iterator)
        return x if lang2 is None or lang1 < lang2 else x[::-1]

    def word_shuffle(self, x, l):
        """
        Randomly shuffle input words.
        """
        if self.params.word_shuffle == 0:
            return x, l

        # define noise word scores
        noise = np.random.uniform(0, self.params.word_shuffle, size=(x.size(0) - 1, x.size(1)))
        noise[0] = -1  # do not move start sentence symbol

        assert self.params.word_shuffle > 1
        x2 = x.clone()
        for i in range(l.size(0)):
            # generate a random permutation
            scores = [i for i in range(l[i] - 1)]
            scores += noise[:l[i] - 1, i]
            permutation = scores.argsort()
            # shuffle words
            x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])
        return x2, l

    def word_dropout(self, x, l):
        """
        Randomly drop input words.
        """
        if self.params.word_dropout == 0:
            return x, l
        assert 0 < self.params.word_dropout < 1

        # define words to drop
        eos = self.params.eos_index
        assert (x[0] == eos).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_dropout
        keep[0] = 1  # do not drop the start sentence symbol

        sentences = []
        lengths = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == eos
            words = x[:l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[j, i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(1, len(words))])
            new_s.append(eos)
            assert len(new_s) >= 3 and new_s[0] == eos and new_s[-1] == eos
            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input
        l2 = torch.LongTensor(lengths)
        x2 = torch.LongTensor(l2.max(), l2.size(0)).fill_(self.params.pad_index)
        for i in range(l2.size(0)):
            x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l2

    def word_blank(self, x, l):
        """
        Randomly blank input words.
        """
        if self.params.word_blank == 0:
            return x, l
        assert 0 < self.params.word_blank < 1

        # define words to blank
        eos = self.params.eos_index
        assert (x[0] == eos).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_blank
        keep[0] = 1  # do not blank the start sentence symbol

        sentences = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == eos
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[j, i] else self.params.mask_index for j, w in enumerate(words)]
            new_s.append(eos)
            assert len(new_s) == l[i] and new_s[0] == eos and new_s[-1] == eos
            sentences.append(new_s)
        # re-construct input
        x2 = torch.LongTensor(l.max(), l.size(0)).fill_(self.params.pad_index)
        for i in range(l.size(0)):
            x2[:l[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l

    def add_noise(self, words, lengths):
        """
        Add noise to the encoder input.
        """
        words, lengths = self.word_shuffle(words, lengths)
        words, lengths = self.word_dropout(words, lengths)
        words, lengths = self.word_blank(words, lengths)
        return words, lengths

    def mask_out(self, x, lengths):
        """
        Decide of random words to mask out, and what target they get assigned.
        """
        params = self.params
        slen, bs = x.size()
        # define target words to predict
        if params.sample_alpha == 0:
            pred_mask = np.random.rand(slen, bs) <= params.word_pred
            pred_mask = torch.from_numpy(pred_mask.astype(np.bool))

        else:
            x_prob = params.mask_scores[x.flatten()]
            n_tgt = math.ceil(params.word_pred * slen * bs)
            tgt_ids = np.random.choice(len(x_prob), n_tgt, replace=False, p=x_prob / x_prob.sum())
            pred_mask = torch.zeros(slen * bs, dtype=torch.bool)
            pred_mask[tgt_ids] = 1
            pred_mask = pred_mask.view(slen, bs)

        # do not predict padding
        pred_mask[x == params.pad_index] = 0
        # this only avoids masking predicting first </s>'s (EOSs) in each sequence
        pred_mask[0] = 0  # TODO: remove

        # mask a number of words == 0 [8] (faster with fp16)
        if params.fp16:
            pred_mask = pred_mask.view(-1)
            n1 = pred_mask.sum().item()
            n2 = max(n1 % 8, 8 * (n1 // 8))
            if n2 != n1:
                pred_mask[torch.nonzero(pred_mask).view(-1)[:n1 - n2]] = 0
            pred_mask = pred_mask.view(slen, bs)
            assert pred_mask.sum().item() % 8 == 0

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_rand = _x_real.clone().random_(params.n_words)
        _x_mask = _x_real.clone().fill_(params.mask_index)
        len_x_real = len(_x_real)

        probs = torch.multinomial(params.word_pred_probs, len_x_real, replacement=True)
        _x = _x_mask * (probs == 0).long() + _x_real * (probs == 1).long() + _x_rand * (probs == 2).long()
        x = x.masked_scatter(pred_mask, _x)
        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask

    def mask_out_image(self, img_boxes, img_feats, img_labels):
        """
        Mask random image features out of a sequence, similar to text counter-part
        """
        # Generate prediction mask
        bs, n_regions, _ = img_feats.size()
        # create a full prob array of shape (bs, n_regions)
        # 0: no change, no prediction
        # 1: multiply the prediction' features with 0 + predict labels (mask)
        # 2: no change at input + predict labels (keep)
        # 3: random features at input + predict labels (rand)
        probs = torch.multinomial(self.params.region_pred_probs, bs * n_regions, replacement=True)
        probs = probs.view(bs, n_regions)

        # positions where there'll be a prediction
        pred_mask = probs > 0
        # ground-truth labels
        y = img_labels[pred_mask]
        # handle randomization
        # shuffle all regions in the batch and pick random ones
        shuf_region_order = torch.randperm(bs * n_regions)
        feat_inventory = img_feats.view(-1, img_feats.size(-1))[shuf_region_order]
        img_feats.masked_scatter_(probs.eq(3).unsqueeze(-1), feat_inventory)

        # mask the features with 0
        # NOTE: We can't mask it with <MASK> at this step because of dimension incompatibility
        # (512 != 1536). So let's make them 0 now and then handle in transformer.py:fwd()
        # if <MASK> is indeed requested to replace regions
        to_mask_positions = probs.eq(1).unsqueeze(-1)
        img_feats.masked_fill_(to_mask_positions, 0.0)

        return img_boxes, img_feats, y, pred_mask, to_mask_positions

    def generate_batch(self, lang1, lang2, name):
        """
        Prepare a batch (for causal or non-causal mode).
        """
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None

        if lang2 is None:
            x, lengths = self.get_batch(name, lang1, stream=True)
            positions = None
            langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
        elif lang1 == lang2:
            (x1, len1) = self.get_batch(name, lang1)
            (x2, len2) = (x1, len1)
            (x1, len1) = self.add_noise(x1, len1)
            x, lengths, positions, langs = concat_batches(x1, len1, lang1_id, x2, len2, lang2_id, params.pad_index,
                                                          params.eos_index, reset_positions=False)
        else:
            (x1, len1), (x2, len2) = self.get_batch(name, lang1, lang2)
            x, lengths, positions, langs = concat_batches(x1, len1, lang1_id, x2, len2, lang2_id, params.pad_index,
                                                          params.eos_index, reset_positions=True)

        return x, lengths, positions, langs, (None, None) if lang2 is None else (len1, len2)

    def generate_batch_vpara(self, lang1, lang2, name):
        """
        Prepare a batch (for causal or non-causal mode).
        """
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None
        img_id = params.lang2id['img']

        if lang2 is not None:
            # VTLM
            _, (boxes, re_feats, gr_feats), (x1, len1), (x2, len2) = self.get_batch_vpara(name, lang1, lang2)
            x, lengths, positions, langs = concat_batches(x1, len1, lang1_id, x2, len2, lang2_id,
                                                          params.pad_index, params.eos_index, reset_positions=True)

            image_langs = torch.empty((params.num_of_regions + 49, len1.size(0))).long().fill_(img_id)

        else:
            # VMLM
            (x, len1), (boxes, re_feats, gr_feats), _ = self.get_batch_vpara(name, lang1, lang2)
            langs = x.clone().fill_(lang1_id)
            image_langs = torch.empty((params.num_of_regions + 49, len1.size(0))).long().fill_(img_id)
            lengths = len1
            positions = None
        return x, lengths, positions, langs, image_langs, boxes, re_feats, gr_feats, (None, None)

    def save_checkpoint(self, name, include_optimizers=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {'epoch': self.epoch, 'n_total_iter': self.n_total_iter, 'best_metrics': self.best_metrics,
                'best_stopping_criterion': self.best_stopping_criterion}

        for name in self.MODEL_NAMES:
            logger.warning(f"Saving {name} parameters ...")
            data[name] = getattr(self, name).state_dict()

        if include_optimizers:
            for name in self.optimizers.keys():
                logger.warning(f"Saving {name} optimizer ...")
                data[f'{name}_optimizer'] = self.optimizers[name].state_dict()

        data['dico_id2word'] = self.data['dico'].id2word
        data['dico_word2id'] = self.data['dico'].word2id
        data['dico_counts'] = self.data['dico'].counts
        data['params'] = {k: v for k, v in self.params.__dict__.items()}

        torch.save(data, path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == '':
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location='cpu')

        # reload model parameters
        for name in self.MODEL_NAMES:
            getattr(self, name).load_state_dict(data[name])

        # reload optimizers
        for name in self.optimizers.keys():
            if False:  # AMP checkpoint reloading is buggy, we cannot do that - TODO: fix - https://github.com/NVIDIA/apex/issues/250
                logger.warning(f"Reloading checkpoint optimizer {name} ...")
                self.optimizers[name].load_state_dict(data[f'{name}_optimizer'])
            else:  # instead, we only reload current iterations / learning rates
                logger.warning(f"Not reloading checkpoint optimizer {name}.")
                for group_id, param_group in enumerate(self.optimizers[name].param_groups):
                    if 'num_updates' not in param_group:
                        logger.warning(f"No 'num_updates' for optimizer {name}.")
                        continue
                    logger.warning(f"Reloading 'num_updates' and 'lr' for optimizer {name}.")
                    param_group['num_updates'] = data[f'{name}_optimizer']['param_groups'][group_id]['num_updates']
                    param_group['lr'] = self.optimizers[name].get_lr_for_step(param_group['num_updates'])

        # reload main metrics
        self.epoch = data['epoch'] + 1
        self.n_total_iter = data['n_total_iter']
        self.best_metrics = data['best_metrics']
        self.best_stopping_criterion = data['best_stopping_criterion']
        logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if self.params.save_periodic > 0 and self.epoch % self.params.save_periodic == 0:
            self.save_checkpoint('checkpoint%i' % self.epoch, include_optimizers=False)

        path = self.params.dump_path
        pth_regexp = re.compile(r"checkpoint(\d+)\.pth")
        files = PathManager.ls(path)
        entries = []
        for i, f in enumerate(files):
            m = pth_regexp.fullmatch(f)
            if m is not None:
                idx = float(m.group(1)) if len(m.groups()) > 0 else i
                entries.append((idx, m.group(0)))
            checkpoint = [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]

        if self.params.keep_best_checkpoints > 0:
            # remove old epoch checkpoints; checkpoints are sorted in descending order
            for old_chk in checkpoint[self.params.keep_best_checkpoints:]:
                if os.path.lexists(old_chk):
                    os.remove(old_chk)
                elif PathManager.exists(old_chk):
                    PathManager.rm(old_chk)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning("Metric \"%s\" not found in scores!" % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_checkpoint('best-%s' % metric, include_optimizers=False)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and (
                self.params.is_master or not self.stopping_criterion[0].endswith('_mt_bleu')
                or not self.stopping_criterion[0].endswith('_mmt_bleu')):
            metric, biggest = self.stopping_criterion

            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            else:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value for more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                if self.params.multi_gpu and 'SLURM_JOB_ID' in os.environ:
                    os.system('scancel ' + os.environ['SLURM_JOB_ID'])
                exit()
        self.save_checkpoint('checkpoint', include_optimizers=True)
        self.epoch += 1

    def round_batch(self, x, lengths, positions, langs):
        """
        For float16 only.
        Sub-sample sentences in a batch, and add padding,
        so that each dimension is a multiple of 8.
        """
        params = self.params
        if not params.fp16 or len(lengths) < 8:
            return x, lengths, positions, langs, None

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            lengths = lengths[idx]
            slen = lengths.max().item()
            x = x[:slen, idx]
            positions = None if positions is None else positions[:slen, idx]
            langs = None if langs is None else langs[:slen, idx]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(0)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            x = torch.cat([x, torch.LongTensor(pad, bs2).fill_(params.pad_index)], 0)
            if positions is not None:
                positions = torch.cat([positions, torch.arange(pad)[:, None] + positions[-1][None] + 1], 0)
            if langs is not None:
                langs = torch.cat([langs, langs[-1][None].expand(pad, bs2)], 0)
            assert x.size() == (ml2, bs2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return x, lengths, positions, langs, idx

    def clm_step(self, lang1, lang2, lambda_coeff):
        """
        Next word prediction step (causal prediction).
        CLM objective.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'decoder'
        model = getattr(self, name)
        model.train()

        # generate batch / select words to predict
        x, lengths, positions, langs, _ = self.generate_batch(lang1, lang2, 'causal')
        x, lengths, positions, langs, _ = self.round_batch(x, lengths, positions, langs)
        alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
        pred_mask = alen[:, None] < lengths[None] - 1
        if params.context_size > 0:  # do not predict without context
            pred_mask[:params.context_size] = 0
        y = x[1:].masked_select(pred_mask[:-1])
        assert pred_mask.sum().item() == y.size(0)

        # cuda
        x, lengths, langs, pred_mask, y = to_cuda(x, lengths, langs, pred_mask, y)

        # forward / loss
        tensor = model('fwd', x=x, lengths=lengths, langs=langs, causal=True)
        _, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats[('CLM-%s' % lang1) if lang2 is None else ('CLM-%s-%s' % (lang1, lang2))].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += lengths.size(0)
        self.stats['processed_w'] += pred_mask.sum().item()

    def mlm_step(self, lang1, lang2, lambda_coeff, iter):
        """
        Masked word prediction step.
        MLM objective is lang2 is None, TLM objective otherwise.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'encoder'
        model = getattr(self, name)
        model.train()

        # generate batch / select words to predict
        x, lengths, positions, langs, _ = self.generate_batch(lang1, lang2, 'pred')
        x, lengths, positions, langs, _ = self.round_batch(x, lengths, positions, langs)
        x, y, pred_mask = self.mask_out(x, lengths)

        # cuda
        x, y, pred_mask, lengths, positions, langs = to_cuda(x, y, pred_mask, lengths, positions, langs)

        # forward / loss
        tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)
        _, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats[('MLM-%s' % lang1) if lang2 is None else ('MLM-%s-%s' % (lang1, lang2))].append(loss.item())
        loss = lambda_coeff * loss

        if iter % 100 == 0:
            self.writer.add_scalar('loss', loss.item(), iter)

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += lengths.size(0)
        self.stats['processed_w'] += pred_mask.sum().item()

    def vmlm_step(self, lang1, lang2, lambda_coeff, iter):
        """
        Masked word prediction step.
        MLM objective is lang2 is None, TLM objective otherwise.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'encoder'
        model = getattr(self, name)
        model.train()

        # generate batch / select words to predict
        x, lengths, positions, langs, image_langs, img_boxes, re_img_feats, gr_img_feats, _ = \
            self.generate_batch_vpara(lang1, lang2, 'pred_object')

        x, lengths, positions, langs, _ = self.round_batch(x, lengths, positions, langs)

        # Get prediction masks
        x, y, txt_pred_mask = self.mask_out(x, lengths)
        # img_boxes, img_feats, img_y, img_pred_mask, img_mask_pos = self.mask_out_image(img_boxes, img_feats,
        # img_labels)

        # cuda

        x, y, txt_pred_mask, lengths, positions, langs, image_langs, img_boxes, re_img_feats, gr_img_feats, = to_cuda(
                x, y, txt_pred_mask, lengths, positions, langs, image_langs, img_boxes, re_img_feats, gr_img_feats)

        # img_y, img_pred_mask, img_mask_pos = to_cuda(img_y, img_pred_mask, img_mask_pos)

        # forward / loss

        tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False,
                       image_langs=image_langs, img_boxes=img_boxes, re_img_feats=re_img_feats,
                       gr_img_feats=gr_img_feats # img_mask_pos=img_mask_pos
                       )

        # slice the outputs back to feed into prediction heads
        if params.visual_first:
            sent_tensor = tensor[params.num_of_regions + 49:]
            img_tensor = tensor[:params.num_of_regions + 49]
        else:
            sent_tensor = tensor[:langs.size(0)]
            img_tensor = tensor[langs.size(0):]

        # bring batch dimension to 0-dim
        img_tensor = img_tensor.permute(1, 0, 2)

        text_scores, text_loss = model('predict', tensor=sent_tensor, pred_mask=txt_pred_mask, y=y, get_scores=True)
        # img_scores, img_loss = model('predict_img_class', tensor=img_tensor, pred_mask=img_pred_mask,
        #                              y=img_y, get_scores=True)


        loss = text_loss  # + img_loss
        text_acc = text_scores.detach().max(1)[1].eq(y).float().mean().item()
        # img_acc = img_scores.detach().max(1)[1].eq(img_y).float().mean().item()

        task = f'VMLM-{lang1}' if lang2 is None else f'VMLM-{lang1}-{lang2}'
        self.stats[(task)].append(loss.item())
        self.stats[(f'{task}-tloss')].append(text_loss.item())
        # self.stats[(f'{task}-vloss')].append(img_loss.item())
        self.stats[(f'{task}-tacc')].append(text_acc)
        # self.stats[(f'{task}-vacc')].append(img_acc)

        # scale it
        loss = lambda_coeff * loss

        if iter % 100 == 0:
            # self.writer.add_scalar('img_loss', img_loss.item(), iter)
            self.writer.add_scalar('text_loss', text_loss.item(), iter)
            self.writer.add_scalar('total_loss', loss.item(), iter)
            # self.writer.add_scalar('img_acc', img_acc, iter)
            self.writer.add_scalar('text_acc', text_acc, iter)

        # optimize

        self.optimize(loss)
        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += lengths.size(0)
        self.stats['processed_w'] += txt_pred_mask.sum().item()  # + img_pred_mask.sum().item()

    def pc_step(self, lang1, lang2, lambda_coeff):
        """
        Parallel classification step. Predict if pairs of sentences are mutual translations of each other.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        name = 'model' if params.encoder_only else 'encoder'
        model = getattr(self, name)
        model.train()

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # sample parallel sentences
        (x1, len1), (x2, len2) = self.get_batch('align', lang1, lang2)
        bs = len1.size(0)
        if bs == 1:  # can happen (although very rarely), which makes the negative loss fail
            self.n_sentences += params.batch_size
            return

        # associate lang1 sentences with their translations, and random lang2 sentences
        y = torch.LongTensor(bs).random_(2)
        idx_pos = torch.arange(bs)
        idx_neg = ((idx_pos + torch.LongTensor(bs).random_(1, bs)) % bs)
        idx = (y == 1).long() * idx_pos + (y == 0).long() * idx_neg
        x2, len2 = x2[:, idx], len2[idx]

        # generate batch / cuda
        x, lengths, positions, langs = concat_batches(x1, len1, lang1_id, x2, len2, lang2_id, params.pad_index,
                                                      params.eos_index, reset_positions=False)
        x, lengths, positions, langs, new_idx = self.round_batch(x, lengths, positions, langs)
        if new_idx is not None:
            y = y[new_idx]
        x, lengths, positions, langs = to_cuda(x, lengths, positions, langs)

        # get sentence embeddings
        h = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)[0]

        # parallel classification loss
        CLF_ID1, CLF_ID2 = 8, 9  # very hacky, use embeddings to make weights for the classifier
        emb = (model.module if params.multi_gpu else model).embeddings.weight
        pred = F.linear(h, emb[CLF_ID1].unsqueeze(0), emb[CLF_ID2, 0])
        loss = F.binary_cross_entropy_with_logits(pred.view(-1), y.to(pred.device).type_as(pred))
        self.stats['PC-%s-%s' % (lang1, lang2)].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += bs
        self.stats['processed_w'] += lengths.sum().item()


class SingleTrainer(Trainer):

    def __init__(self, model, data, params):
        self.MODEL_NAMES = ['model']

        # model / data / params
        self.model = model
        self.data = data
        self.params = params

        super().__init__(data, params)


class EncDecTrainer(Trainer):

    def __init__(self, encoder, decoder, data, params):

        self.MODEL_NAMES = ['encoder', 'decoder']

        # model / data / params
        self.encoder = encoder
        self.decoder = decoder
        self.data = data
        self.dim = params.emb_dim
        self.params = params
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.dico = data['dico']
        self.pred_layer = PredLayer(params)

        super().__init__(data, params)

    def mask_word(self, w):
        _w_real = w
        _w_rand = np.random.randint(self.params.n_words, size=w.shape)
        _w_mask = np.full(w.shape, self.params.mask_index)

        probs = torch.multinomial(self.params.word_pred_probs, len(_w_real), replacement=True)

        _w = _w_mask * (probs == 0).numpy() + _w_real * (probs == 1).numpy() + _w_rand * (probs == 2).numpy()
        return _w

    def unfold_segments(self, segs):
        """Unfold the random mask segments, for example:
           The shuffle segment is [2, 0, 0, 2, 0],
           so the masked segment is like:
           [1, 1, 0, 0, 1, 1, 0]
           [1, 2, 3, 4, 5, 6, 7] (positions)
           (1 means this token will be masked, otherwise not)
           We return the position of the masked tokens like:
           [1, 2, 5, 6]
        """
        pos = []
        curr = 1  # We do not mask the start token
        for l in segs:
            if l >= 1:
                pos.extend([curr + i for i in range(l)])
                curr += l
            else:
                curr += 1
        return np.array(pos)

    def shuffle_segments(self, segs, unmasked_tokens):
        """
        We control 20% mask segment is at the start of sentences
                   20% mask segment is at the end   of sentences
                   60% mask segment is at random positions,
        """

        p = np.random.random()
        if p >= 0.8:
            shuf_segs = segs[1:] + unmasked_tokens
        elif p >= 0.6:
            shuf_segs = segs[:-1] + unmasked_tokens
        else:
            shuf_segs = segs + unmasked_tokens

        random.shuffle(shuf_segs)

        if p >= 0.8:
            shuf_segs = segs[0:1] + shuf_segs
        elif p >= 0.6:
            shuf_segs = shuf_segs + segs[-1:]
        return shuf_segs

    def get_segments(self, mask_len, span_len):
        segs = []
        while mask_len >= span_len:
            segs.append(span_len)
            mask_len -= span_len
        if mask_len != 0:
            segs.append(mask_len)
        return segs

    def restricted_mask_sent(self, x, l, span_len=100000):
        """ Restricted mask sents
            if span_len is equal to 1, it can be viewed as
            discrete mask;
            if span_len -> inf, it can be viewed as
            pure sentence mask
        """
        if span_len <= 0:
            span_len = 1
        max_len = 0
        positions, inputs, targets, outputs, = [], [], [], []
        mask_len = round(len(x[:, 0]) * self.params.word_mass)
        len2 = [mask_len for i in range(l.size(0))]

        unmasked_tokens = [0 for i in range(l[0] - mask_len - 1)]
        segs = self.get_segments(mask_len, span_len)

        for i in range(l.size(0)):
            words = np.array(x[:l[i], i].tolist())
            shuf_segs = self.shuffle_segments(segs, unmasked_tokens)
            pos_i = self.unfold_segments(shuf_segs)
            output_i = words[pos_i].copy()
            target_i = words[pos_i - 1].copy()
            words[pos_i] = self.mask_word(words[pos_i])

            inputs.append(words)
            targets.append(target_i)
            outputs.append(output_i)
            positions.append(pos_i - 1)

        x1 = torch.LongTensor(max(l), l.size(0)).fill_(self.params.pad_index)
        x2 = torch.LongTensor(mask_len, l.size(0)).fill_(self.params.pad_index)
        y = torch.LongTensor(mask_len, l.size(0)).fill_(self.params.pad_index)
        pos = torch.LongTensor(mask_len, l.size(0)).fill_(self.params.pad_index)
        l1 = l.clone()
        l2 = torch.LongTensor(len2)
        for i in range(l.size(0)):
            x1[:l1[i], i].copy_(torch.LongTensor(inputs[i]))
            x2[:l2[i], i].copy_(torch.LongTensor(targets[i]))
            y[:l2[i], i].copy_(torch.LongTensor(outputs[i]))
            pos[:l2[i], i].copy_(torch.LongTensor(positions[i]))

        pred_mask = y != self.params.pad_index
        y = y.masked_select(pred_mask)
        return x1, l1, x2, l2, y, pred_mask, pos

    def mt_step(self, lang1, lang2, lambda_coeff):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        self.encoder.train()
        self.decoder.train()

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # generate batch
        if lang1 == lang2:
            (x1, len1) = self.get_batch('ae', lang1)
            (x2, len2) = (x1, len1)
            (x1, len1) = self.add_noise(x1, len1)
        else:
            (x1, len1), (x2, len2) = self.get_batch('mt', lang1, lang2)
        langs1 = x1.clone().fill_(lang1_id)
        langs2 = x2.clone().fill_(lang2_id)

        # target words to predict
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
        y = x2[1:].masked_select(pred_mask[:-1])
        assert len(y) == (len2 - 1).sum().item()

        # cuda
        x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

        # encode source sentence
        enc1 = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
        enc1 = enc1.transpose(0, 1)

        # decode target sentence
        dec2 = self.decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)

        # loss
        _, loss = self.decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats[('AE-%s' % lang1) if lang1 == lang2 else ('MT-%s-%s' % (lang1, lang2))].append(loss.item())
        loss = lambda_coeff * loss
        # tensor(num, device='cuda:0', grad_fn=<MulBackward0>)
        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += (len2 - 1).sum().item()


    def vae_step(self, lang1, lang2, lambda_coeff):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        self.encoder.train()
        self.decoder.train()

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        img_id = params.lang2id['img']

        # generate batch
        if lang1 == lang2:
            (x1, len1), (img_boxes, re_img_feats, gr_img_feats, gl_img_feats), _ = self.get_batch_vpara(
                'vae', lang1)
            (x2, len2) = (x1, len1)
            (x1, len1) = self.add_noise(x1, len1)
        else:
            if lang1 > lang2:
                _, (img_boxes, re_img_feats, gr_img_feats, gl_img_feats), (x1, len1), (x2, len2) = (
                    self.get_batch('mmt', lang1, lang2))
            else:
                (x1, len1), (x2, len2), (img_boxes, re_img_feats, gr_img_feats, gl_img_feats), _ = self.get_batch(
                    'mmt', lang1, lang2)

        langs1 = x1.clone().fill_(lang1_id)
        langs2 = x2.clone().fill_(lang2_id)


        # target words to predict
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
        y = x2[1:].masked_select(pred_mask[:-1])
        assert len(y) == (len2 - 1).sum().item()

        if params.re_img_feats:
            src_len = len1 + params.num_of_regions
            img_langs = torch.empty((params.num_of_regions, langs1.size(1))).long().fill_(img_id)
        elif params.gr_img_feats:
            src_len = len1 + 49
            img_langs = torch.empty((49, langs1.size(1))).long().fill_(img_id)
        elif params.gl_img_feats:
            src_len = len1 + 1
            img_langs = torch.empty((1, langs1.size(1))).long().fill_(img_id)
        elif params.re_img_feats and params.gr_img_feats:
            src_len = len1 + params.num_of_regions + 49
            img_langs = torch.empty((params.num_of_regions + 49, langs1.size(1))).long().fill_(img_id)
        elif params.re_img_feats and params.gl_img_feats:
            src_len = len1 + params.num_of_regions + 1
            img_langs = torch.empty((params.num_of_regions + 1, langs1.size(1))).long().fill_(img_id)
        elif params.gr_img_feats and params.gl_img_feats:
            src_len = len1 + 49 + 1
            img_langs = torch.empty((49 + 1, langs1.size(1))).long().fill_(img_id)
        elif params.re_img_feats and params.gr_img_feats and params.gl_img_feats:
            src_len = len1 + params.num_of_regions + 49 + 1
            img_langs = torch.empty((params.num_of_regions + 49 + 1, langs1.size(1))).long().fill_(img_id)
        else:
            raise Exception("There is no other case")
        print('00000000000000', re_img_feats)
        # cuda
        x1, len1, langs1, x2, len2, langs2, y, img_boxes, img_langs, re_img_feats, gr_img_feats, gl_img_feats, src_len \
            = (to_cuda(x1, len1, langs1, x2, len2, langs2, y, img_boxes, img_langs, re_img_feats, gr_img_feats,
                       gl_img_feats, src_len))


        # encode source sentence
        enc1 = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False, image_langs=img_langs,
                            img_boxes=img_boxes, re_img_feats=re_img_feats, gr_img_feats=gr_img_feats,
                            gl_img_feats=gl_img_feats)

        enc1 = enc1.transpose(0, 1)

        # decode target sentence
        dec2 = self.decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=src_len)

        # loss
        _, loss = self.decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats[('VAE-%s' % lang1) if lang1 == lang2 else ('MMT-%s-%s' % (lang1, lang2))].append(loss.item())

        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += (len2 - 1).sum().item()


    def mmt_step(self, lang1, lang2, lambda_coeff):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        self.encoder.train()
        self.decoder.train()

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        img_id = params.lang2id['img']

        # generate batch
        if lang1 == lang2:
            (x1, len1), (img_boxes, re_img_feats, gr_img_feats, gl_img_feats), _ = self.get_batch_vpara('vae', lang1)
            (x2, len2) = (x1, len1)
            (x1, len1) = self.add_noise(x1, len1)
        else:
            if lang1 > lang2:
                _, (img_boxes, re_img_feats, gr_img_feats, gl_img_feats), (x1, len1), (x2, len2) = self.get_batch('mmt', lang1, lang2)
            else:
                (x1, len1), (x2, len2), (img_boxes, re_img_feats, gr_img_feats, gl_img_feats), _ = self.get_batch('mmt', lang1, lang2)

        langs1 = x1.clone().fill_(lang1_id)
        langs2 = x2.clone().fill_(lang2_id)
        img_langs = torch.empty((params.num_of_regions + 49, langs1.size(1))).long().fill_(img_id)

        # target words to predict
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
        y = x2[1:].masked_select(pred_mask[:-1])
        assert len(y) == (len2 - 1).sum().item()

        # cuda

        x1, len1, langs1, x2, len2, langs2, y, img_langs, img_boxes, re_img_feats, gr_img_feats = to_cuda(
            x1, len1, langs1, x2, len2, langs2, y, img_langs, img_boxes, re_img_feats, gr_img_feats)


        # encode source sentence
        enc1 = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False, image_langs=img_langs,
                            img_boxes=img_boxes, re_img_feats=re_img_feats, gr_img_feats=gr_img_feats)

        enc1 = enc1.transpose(0, 1)

        src_len = len1
        if params.inputs_concat:
            src_len = len1 + params.num_of_regions + 49

        # decode target sentence
        dec2 = self.decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=src_len)

        # loss
        _, loss = self.decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats[('VAE-%s' % lang1) if lang1 == lang2 else ('MMT-%s-%s' % (lang1, lang2))].append(loss.item())

        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += (len2 - 1).sum().item()


        self.stats[('RL-%s-%s' % (lang1, lang2))].append(loss.item())
        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += (len2 - 1).sum().item()

    def bt_step(self, lang1, lang2, lang3, lambda_coeff):
        """
        Back-translation step for machine translation.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        assert lang1 == lang3 and lang1 != lang2 and lang2 is not None
        params = self.params
        _encoder = self.encoder.module if params.multi_gpu else self.encoder
        _decoder = self.decoder.module if params.multi_gpu else self.decoder

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # generate source batch
        x1, len1 = self.get_batch('bt', lang1)
        langs1 = x1.clone().fill_(lang1_id)

        # cuda
        x1, len1, langs1 = to_cuda(x1, len1, langs1)

        # generate a translation
        with torch.no_grad():
            # evaluation mode
            self.encoder.eval()
            self.decoder.eval()

            # encode source sentence and translate it
            enc1 = _encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            x2, len2 = _decoder.generate(enc1, len1, lang2_id, max_len=int(1.3 * len1.max().item() + 5))
            langs2 = x2.clone().fill_(lang2_id)

            # free CUDA memory
            del enc1

            # training mode
            self.encoder.train()
            self.decoder.train()

        # encode generate sentence
        enc2 = self.encoder('fwd', x=x2, lengths=len2, langs=langs2, causal=False)
        enc2 = enc2.transpose(0, 1)

        # words to predict
        alen = torch.arange(len1.max(), dtype=torch.long, device=len1.device)
        pred_mask = alen[:, None] < len1[None] - 1  # do not predict anything given the last target word
        y1 = x1[1:].masked_select(pred_mask[:-1])

        # decode original sentence
        dec3 = self.decoder('fwd', x=x1, lengths=len1, langs=langs1, causal=True, src_enc=enc2, src_len=len2)

        # loss
        _, loss = self.decoder('predict', tensor=dec3, pred_mask=pred_mask, y=y1, get_scores=False)
        self.stats[('BT-%s-%s-%s' % (lang1, lang2, lang3))].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len1.size(0)
        self.stats['processed_w'] += (len1 - 1).sum().item()

    def vbt_step(self, lang1, lang2, lang3, lambda_coeff):
        """
        Back-translation step for machine translation.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        assert lang1 == lang3 and lang1 != lang2 and lang2 is not None
        params = self.params
        _encoder = self.encoder.module if params.multi_gpu else self.encoder
        _decoder = self.decoder.module if params.multi_gpu else self.decoder

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        img_id = params.lang2id['img']

        # generate source batch
        (x1, len1), (img_boxes, re_img_feats, gr_img_feats, gl_img_feats), _ = self.get_batch_vpara('vbt', lang1)

        langs1 = x1.clone().fill_(lang1_id)

        if params.re_img_feats:
            src_len = len1 + params.num_of_regions
        elif params.gr_img_feats:
            src_len = len1 + 49
            img_langs = torch.empty((49, langs1.size(1))).long().fill_(img_id)
        elif params.gl_img_feats:
            src_len = len1 + 1
            img_langs = torch.empty((1, langs1.size(1))).long().fill_(img_id)
        elif params.re_img_feats and params.gr_img_feats:
            src_len = len1 + params.num_of_regions + 49
            img_langs = torch.empty((params.num_of_regions + 49, langs1.size(1))).long().fill_(img_id)
        elif params.re_img_feats and params.gl_img_feats:
            src_len = len1 + params.num_of_regions + 1
            img_langs = torch.empty((params.num_of_regions + 1, langs1.size(1))).long().fill_(img_id)
        elif params.gr_img_feats and params.gl_img_feats:
            src_len = len1 + 49 + 1
            img_langs = torch.empty((49 + 1, langs1.size(1))).long().fill_(img_id)
        elif params.re_img_feats and params.gr_img_feats and params.gl_img_feats:
            src_len = len1 + params.num_of_regions + 49 + 1
            img_langs = torch.empty((params.num_of_regions + 49 + 1, langs1.size(1))).long().fill_(img_id)
        else:
            raise Exception("There is no other case")

        # cuda

        x1, len1, langs1, img_langs, img_boxes, re_img_feats, gr_img_feats, gl_img_feats, src_len = to_cuda(
            x1, len1, langs1, img_langs,img_boxes, re_img_feats, gr_img_feats, gl_img_feats, src_len)

        # generate a translation
        with torch.no_grad():
            # evaluation mode
            self.encoder.eval()
            self.decoder.eval()

            # encode source sentence and translate it
            enc1 = _encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False, image_langs=img_langs,
                            img_boxes=img_boxes, re_img_feats=re_img_feats, gr_img_feats=gr_img_feats,
                            gl_img_feats=gl_img_feats)

            enc1 = enc1.transpose(0, 1)
            src_len1 = len1
            if params.inputs_concat:
                src_len1 = len1 + params.num_of_regions + 49
            x2, len2 = _decoder.generate(enc1, src_len1, lang2_id, max_len=int(1.3 * len1.max().item() + 5))
            langs2 = x2.clone().fill_(lang2_id)

            # free CUDA memory
            del enc1

            # training mode
            self.encoder.train()
            self.decoder.train()

        # encode generate sentence
        enc2 = self.encoder('fwd', x=x2, lengths=len2, langs=langs2, causal=False, image_langs=img_langs,
                            img_boxes=img_boxes, re_img_feats=re_img_feats, gr_img_feats=gr_img_feats,
                            gl_img_feats=gl_img_feats)

        enc2 = enc2.transpose(0, 1)

        # words to predict
        alen = torch.arange(len1.max(), dtype=torch.long, device=len1.device)
        pred_mask = alen[:, None] < len1[None] - 1  # do not predict anything given the last target word
        y1 = x1[1:].masked_select(pred_mask[:-1])

        # decode original sentence
        dec3 = self.decoder('fwd', x=x1, lengths=len1, langs=langs1, causal=True, src_enc=enc2, src_len=src_len)

        # loss
        word_scores, loss = self.decoder('predict', tensor=dec3, pred_mask=pred_mask, y=y1, get_scores=False)
        self.stats[('VBT-%s-%s-%s' % (lang1, lang2, lang3))].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len1.size(0)
        self.stats['processed_w'] += (len1 - 1).sum().item()


    def mass_step(self, lang, lambda_coeff):

        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        self.encoder.train()
        self.decoder.train()

        lang1_id = params.lang2id[lang]
        lang2_id = params.lang2id[lang]
        x_, len_ = self.get_batch('mass', lang)

        (x1, len1, x2, len2, y, pred_mask, positions) = self.restricted_mask_sent(x_, len_, int(params.lambda_span))

        langs1 = x1.clone().fill_(lang1_id)
        langs2 = x2.clone().fill_(lang2_id)

        x1, len1, langs1, x2, len2, langs2, y, positions = to_cuda(x1, len1, langs1, x2, len2, langs2, y, positions)

        enc1 = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
        enc1 = enc1.transpose(0, 1)

        enc_mask = x1.ne(params.mask_index)
        enc_mask = enc_mask.transpose(0, 1)

        dec2 = self.decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True,
                            src_enc=enc1, src_len=len1, positions=positions, enc_mask=enc_mask)

        _, loss = self.decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats[('MASS-%s' % lang)].append(loss.item())

        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += (len2 - 1).sum().item()

    def vmass_step(self, lang, lambda_coeff):

        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        self.encoder.train()
        self.decoder.train()

        lang1_id = params.lang2id[lang]
        lang2_id = params.lang2id[lang]
        img_id = params.lang2id['img']

        (x_, len_), (img_boxes, img_feats, img_labels), _ = self.get_batch_vpara('vmass', lang)
        (x1, len1, x2, len2, y, pred_mask, positions) = self.restricted_mask_sent(x_, len_, int(params.lambda_span))

        langs1 = x1.clone().fill_(lang1_id)
        langs2 = x2.clone().fill_(lang2_id)
        img_langs = torch.empty((params.num_of_regions, langs1.size(1))).long().fill_(img_id)

        x1, len1, langs1, x2, len2, langs2, y, positions = to_cuda(x1, len1, langs1, x2, len2, langs2, y, positions)
        img_boxes, img_feats, img_langs, = to_cuda(img_boxes, img_feats, img_langs)

        enc1 = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False, image_langs=img_langs,
                            img_boxes=img_boxes, img_feats=img_feats)

        enc1 = enc1.transpose(0, 1)
        img_mask = x1.new_ones((params.num_of_regions, x1.size(1)))
        src_len = len1

        if params.inputs_concat:
            src_len = len1 + params.num_of_regions
            if params.visual_first:
                x1 = torch.cat((img_mask, x1), dim=0)
            else:
                x1 = torch.cat((x1, img_mask), dim=0)

        enc_mask = x1.ne(params.mask_index)
        enc_mask = enc_mask.transpose(0, 1)
        dec2 = self.decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True,
                            src_enc=enc1, src_len=src_len, positions=positions, enc_mask=enc_mask)

        _, loss = self.decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
        self.stats[('VMASS-%s' % lang)].append(loss.item())

        self.optimize(loss)

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += (len2 - 1).sum().item()


def convert_to_text(batch, lengths, dico, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    # assert lengths.max() == slen and lengths.shape[0] == bs
    # assert (batch[0] == params.eos_index).sum() == bs
    # assert (batch == params.eos_index).sum() == 2 * bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    return sentences


def bleu(hypotheses, references, tokenize="13a"):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param tokenize: one of {'none', '13a', 'intl', 'zh', 'ja-mecab'}
    :return:
    """
    return sacrebleu.corpus_bleu(hypotheses=hypotheses, references=[references], tokenize=tokenize).score