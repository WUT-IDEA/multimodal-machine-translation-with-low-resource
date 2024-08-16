# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoder
from fairseq.modules import (FairseqDropout, LayerDropModuleList, LayerNorm, MultiheadAttention,
                             PositionalEmbedding, SinusoidalPositionalEmbedding)
from fairseq.modules import transformer_layer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from fairseq.models.transformer import (TransformerConfig)
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerEncoderBase":
        return "TransformerEncoder"
    else:
        return module_name


class Imgprojector(nn.Module):
    """Regional image projection layer"""

    def __init__(self, cfg, embed_dim):
        super().__init__()
        self.relu = getattr(cfg, 'reg_img_relu', True)
        self.linear = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x):
        x = self.linear(x)
        return F.relu(x) if self.relu else x


class RGANNEncoderBase(FairseqEncoder):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(cfg.dropout,
                                             module_name=module_name_fordropout(self.__class__.__name__))
        self.encoder_layerdrop = cfg.encoder.layerdrop
        self.return_fc = return_fc

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = cfg.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (PositionalEmbedding(cfg.max_source_positions, embed_dim,
                                                    self.padding_idx,
                                                    learned=cfg.encoder.learned_pos)
                                if not cfg.no_token_positional_embeddings else None)
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(nn.Linear(embed_dim, embed_dim, bias=False),
                                                  cfg.quant_noise.pq,
                                                  cfg.quant_noise.pq_block_size)
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([self.build_encoder_layer(cfg) for i in range(cfg.encoder.layers)])
        self.num_layers = len(self.layers)

        if cfg.encoder.normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

        # visual attention
        self.region_img_project = Imgprojector(cfg, embed_dim)
        self.grid_img_project = Imgprojector(cfg, embed_dim)
        self.visual_attn = MultiheadAttention(embed_dim, cfg.encoder.attention_heads,
                                              dropout=cfg.attention_dropout, self_attention=True)

        self.merge_attn = AttackMultiheadAttention(embed_dim, cfg.encoder.attention_heads,
                                                   kdim=getattr(cfg, "encoder_embed_dim", None),
                                                   vdim=getattr(cfg, "encoder_embed_dim", None),
                                                   dropout=cfg.attention_dropout,
                                                   encoder_decoder_attention=True)

        self.gate = nn.Linear(2 * embed_dim, embed_dim)

        # calibration
        self.build_attacker(cfg)
        self.weight = 1 * math.exp(-150000 / 10 ** 5)

    def set_weight(self, num_updates):
        self.weight = 1 * math.exp(-num_updates / 10 ** 5)

    def build_attacker(self, args):
        self.attacker = Attack(embed_dim=args.decoder_embed_dim, num_heads=args.decoder_attention_heads,
                               dropout=args.attention_dropout)

    def build_encoder_layer(self, cfg):
        layer = transformer_layer.TransformerEncoderLayerBase(cfg, return_fc=self.return_fc)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(self, src_tokens, token_embedding: Optional[torch.Tensor] = None):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(self, src_tokens, src_lengths: Optional[torch.Tensor] = None,
                return_all_hiddens: bool = False, grid_img_features=None, region_img_features=None,
                token_embeddings: Optional[torch.Tensor] = None, attacker=None, loss_type: str = 'nmt',
                new_weight: float = 1.0):

        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(src_tokens, src_lengths, return_all_hiddens,
                                       token_embeddings, grid_img_features, region_img_features,
                                       loss_type, new_weight)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(self, src_tokens, src_lengths: Optional[torch.Tensor] = None,
                           return_all_hiddens: bool = False, token_embeddings: Optional[torch.Tensor] = None,
                           grid_img_features=None, region_img_features=None,
                           loss_type='nmt', new_weight=1.0):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            lr = layer(x, encoder_padding_mask=encoder_padding_mask if has_pads else None)

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        # visual attention

        grid_img_feat = self.grid_img_project(grid_img_features.type_as(x))  # [bsz, 49, dim]
        grid_img_feat = grid_img_feat.transpose(0, 1)  # [49, bsz, dim]

        region_img_feat = self.region_img_project(region_img_features.type_as(x))  # [bsz, img_len, dim]
        region_img_feat = region_img_feat.transpose(0, 1)  # [img_len, bsz, dim]

        region_img_mask = torch.sum(region_img_feat, dim=-1).eq(0).transpose(0, 1)
        grid_img_mask = torch.sum(grid_img_feat, dim=-1).eq(1).transpose(0, 1)

        # visual attention
        # [img_len, bsz, dim]
        img_feat, _ = self.visual_attn(query=region_img_feat, key=grid_img_feat, value=grid_img_feat,
                                       key_padding_mask=grid_img_mask)

        mask = self.attacker(x, img_feat, img_feat, region_img_mask)

        merge, _ = self.merge_attn(query=x, key=img_feat, value=img_feat, key_padding_mask=region_img_mask,
                                   static_kv=True, mask=mask,
                                   loss_type=loss_type, new_weight=new_weight)

        concat = torch.cat((merge, x), dim=-1)
        gated = torch.sigmoid(self.gate(concat))

        fuse = (1 - gated) * x + gated * merge

        x = fuse + x

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (src_tokens.ne(self.padding_idx).sum(dim=1, dtype=torch.int32).reshape(-1, 1)
                       .contiguous())
        return {"encoder_out": [x],  # T x B x C
                "encoder_padding_mask": [encoder_padding_mask],  # B x T
                "encoder_embedding": [encoder_embedding],  # B x T x C
                "encoder_states": encoder_states,  # List[T x B x C]
                "fc_results": fc_results,  # List[T x B x C]
                "src_tokens": [],
                "src_lengths": [src_lengths]}

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [encoder_out["encoder_padding_mask"][0].index_select(0, new_order)]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [encoder_out["encoder_embedding"][0].index_select(0, new_order)]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {"encoder_out": new_encoder_out,  # T x B x C
                "encoder_padding_mask": new_encoder_padding_mask,  # B x T
                "encoder_embedding": new_encoder_embedding,  # B x T x C
                "encoder_states": encoder_states,  # List[T x B x C]
                "src_tokens": src_tokens,  # B x T
                "src_lengths": src_lengths,  # B x 1
                }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict["{}.embed_positions._float_tensor".format(name)] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class RGANNEncoder(RGANNEncoderBase):
    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        self.args = args
        super().__init__(TransformerConfig.from_namespace(args), dictionary, embed_tokens,
                         return_fc=return_fc)

    def build_encoder_layer(self, args):
        return super().build_encoder_layer(TransformerConfig.from_namespace(args))


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class AttackMultiheadAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout, kdim=None, vdim=None, encoder_decoder_attention=False):
        super().__init__(embed_dim, num_heads, dropout=dropout, kdim=kdim, vdim=vdim,
                         encoder_decoder_attention=encoder_decoder_attention)

    def forward(self, query, key: Optional[Tensor], value: Optional[Tensor],
                key_padding_mask: Optional[Tensor] = None,
                incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                need_weights: bool = True, static_kv: bool = False,
                attn_mask: Optional[Tensor] = None, before_softmax: bool = False,
                need_head_weights: bool = False, mask=None, loss_type: str = 'nmt',
                new_weight: float = 1.0) -> Tuple[Tensor, Optional[Tensor]]:

        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if (not self.onnx_trace and not not is_tpu  # don't use PyTorch version on TPUs
                and incremental_state is None and not static_kv
                # A workaround for quantization to work. Otherwise JIT compilation
                # treats bias in linear module as method.
                and not torch.jit.is_scripting()):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, torch.empty([0]),
                                                  torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                                                  self.bias_k, self.bias_v, self.add_zero_attn, self.dropout_module.p,
                                                  self.out_proj.weight, self.out_proj.bias,
                                                  self.training or self.dropout_module.apply_during_inference,
                                                  key_padding_mask, need_weights, attn_mask,
                                                  use_separate_proj_weight=True, q_proj_weight=self.q_proj.weight,
                                                  k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight)

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask,
                                              key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = (q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1))
        if k is not None:
            k = (k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1))
        if v is not None:
            v = (v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1))

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask, prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz, src_len=k.size(1), static_kv=static_kv)

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        
        # masked perturbation
        if loss_type == 'nmt':
            attn_weights_float = new_weight * attn_weights_float + (1 - new_weight) * torch.mul(attn_weights_float,
                                                                                                torch.exp(1 - mask))
        elif loss_type == 'mask':
            # attn_weights_float = torch.mul(attn_weights, mask)
            attn_weights_float = torch.add(torch.mul(attn_weights_float, mask),
                                           torch.mul(torch.mean(attn_weights_float, -1, True), 1 - mask))

        # tmp=attn_weights_float
        # if key_padding_mask is not None:
        #     attn_weights_float = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len)
        #     attn_weights_float = attn_weights_float.masked_fill(
        #             key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
        #             float("-inf")
        #         )
        #     attn_weights_float = attn_weights_float.view(bsz * self.num_heads, tgt_len, src_len)

        # attn_weights_float = utils.softmax(attn_weights_float, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
                # attn_weights = attn_weights[0]

        return attn, attn_weights


class Attack(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout, kdim=None, vdim=None):
        super().__init__(embed_dim, num_heads, kdim=kdim, vdim=vdim, dropout=dropout)

    def forward(self, query, key: Optional[Tensor], value: Optional[Tensor],
                key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:

        tgt_len, bsz, embed_dim = query.size()

        is_tpu = query.device.type == "xla"
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(key)

        q *= self.scaling
        q = (q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1))
        k = (k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1))
        v = (v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1))

        src_len = k.size(1)

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = torch.sigmoid(attn_weights)
        # attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)

        return attn_weights
