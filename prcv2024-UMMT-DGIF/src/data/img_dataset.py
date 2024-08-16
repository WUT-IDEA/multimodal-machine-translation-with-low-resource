from logging import getLogger
import numpy as np
import torch
import os
import pickle


from .dataset import ParallelDataset, Dataset


logger = getLogger()


def load_images(sentence_ids, re_feat_path, gr_feat_path, gl_feat_path, img_names, n_regions):
    gr_img_feats, img_boxes, re_img_feats, img_labels, gl_img_feats = [], [], [], [], []

    for idx in sentence_ids:
        # Everything should be loadable. If features do not exist
        # use the dummy empty_feats.pkl

        f_name = os.path.join(re_feat_path, img_names[idx])
        with open(f_name, "rb") as f:
            x = pickle.load(f)
            assert len(x) != 0 and len(x["detection_scores"]) == 36

            # reduce to requested # of regions
            img_boxes.append(x['detection_boxes'][:n_regions].squeeze())
            re_img_feats.append(x['detection_features'][:n_regions].squeeze())
            img_labels.append(x['detection_classes'][:n_regions].squeeze())

        f_name = os.path.join(gr_feat_path, img_names[idx])
        with open(f_name, "rb") as f:
            x = pickle.load(f)
            x = torch.tensor(x)
            x = x.view(x.size(0) * x.size(1), x.size(2))
            x = x.numpy()
            gr_img_feats.append(x)

        f_name = os.path.join(gl_feat_path, img_names[idx])
        with open(f_name, "rb") as f:
            x = pickle.load(f)
            x = x['img_feat']
            x = torch.tensor(x)
            x = x.numpy()
            gl_img_feats.append(x)

    # convert to numpy arrays
    # detection_scores is not used anywhere so we don't return it
    img_boxes = torch.from_numpy(np.array(img_boxes, dtype=img_boxes[0].dtype))
    re_img_feats = torch.from_numpy(np.array(re_img_feats, dtype=re_img_feats[0].dtype))
    img_labels = torch.from_numpy(np.array(img_labels, dtype='int64'))
    gr_img_feats = torch.from_numpy(np.array(gr_img_feats, dtype=gr_img_feats[0].dtype)).to(torch.float32)
    gl_img_feats = torch.from_numpy(np.array(gl_img_feats, dtype=gl_img_feats[0].dtype)).to(torch.float32)

    return img_boxes, re_img_feats, gr_img_feats, gl_img_feats


class DatasetWithRegions(Dataset):
    def __init__(self, sent, pos, image_names, params):
        super().__init__(sent, pos, params)
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.sent = sent
        self.pos = pos
        self.lengths = self.pos[:, 1] - self.pos[:, 0]

        self.num_of_regions = params.num_of_regions
        self.re_features_path = params.region_feats_path
        self.gr_features_path = params.grid_feats_path
        self.gl_features_path = params.global_feats_path
        self.image_names = np.array(image_names)

        # check number of sentences
        assert len(self.pos) == (self.sent == self.eos_index).sum()

        # Set RNG
        self._rng = np.random.RandomState(seed=params.iter_seed)

        # sanity checks
        self.check()

    def remove_long_sentences(self, max_len):
        indices = super().remove_long_sentences(max_len)
        self.image_names = self.image_names[indices]
        self.check()

    def select_data(self, a, b):
        super().select_data(a, b)
        self.image_names = self.image_names[a:b]

    def load_images(self, sentence_ids):
        return load_images(sentence_ids, self.re_features_path, self.gr_features_path, self.gl_features_path,
                           self.image_names, self.num_of_regions)

    def get_batches_iterator(self, batches, return_indices):
        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                self._rng.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]

            pos = self.pos[sentence_ids]
            sent = self.batch_sentences([self.sent[a:b] for a, b in pos])

            # Visual features dictionary
            img_boxes, re_img_feats, gr_img_feats, gl_img_feats = self.load_images(sentence_ids)

            yield (sent, (img_boxes, re_img_feats, gr_img_feats, gl_img_feats), sentence_ids)


class ParallelDatasetWithRegions(ParallelDataset):
    def __init__(self, sent1, pos1, sent2, pos2, image_names, params):
        super().__init__(sent1, pos1, sent2, pos2, params)
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size
        self.sent1 = sent1
        self.sent2 = sent2
        self.pos1 = pos1
        self.pos2 = pos2
        self.image_names = np.array(image_names)
        self.re_features_path = params.region_feats_path
        self.gr_features_path = params.grid_feats_path
        self.gl_features_path = params.global_feats_path
        self.num_of_regions = params.num_of_regions
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]

        # Set RNG
        self._rng = np.random.RandomState(seed=params.iter_seed)

        # sanity checks
        self.check()

    def remove_long_sentences(self, max_len):
        indices = super().remove_long_sentences(max_len)
        self.image_names = self.image_names[indices]

    def select_data(self, a, b):
        super().select_data(a, b)
        self.image_names = self.image_names[a:b]

    def load_images(self, sentence_ids):
        return load_images(sentence_ids, self.re_features_path, self.gr_features_path, self.gl_features_path,
                           self.image_names, self.num_of_regions)

    def get_batches_iterator(self, batches, return_indices):
        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                self._rng.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]

            # Textual stream
            pos1 = self.pos1[sentence_ids]
            pos2 = self.pos2[sentence_ids]
            sent1 = self.batch_sentences([self.sent1[a:b] for a, b in pos1])
            sent2 = self.batch_sentences([self.sent2[a:b] for a, b in pos2])

            # Visual features as separate tensors
            img_boxes, re_img_feats, gr_img_feats, gl_img_feats = self.load_images(sentence_ids)

            yield (sent1, sent2, (img_boxes, re_img_feats, gr_img_feats, gl_img_feats), sentence_ids)
