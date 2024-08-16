# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple
from fairseq.models import register_model, register_model_architecture
from .encoder import RGMMTEncoder
from fairseq.models.transformer import (TransformerModel, base_architecture, TransformerDecoder,
                                        TransformerConfig)
from dataclasses import dataclass, field
from fairseq.dataclass import ChoiceEnum, FairseqDataclass


@dataclass
class RGMMTConfig(TransformerConfig):
    img_feature_dim: int = field(default=512, metadata={"help": "image features dimension"})


@register_model("rgmmt_transformer", dataclass=RGMMTConfig)
class RGMMTTModel(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg: RGMMTConfig, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        self.cfg = cfg

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        model = super().build_model(cfg, task)

        return model

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return RGMMTEncoder(cfg, src_dict, embed_tokens)

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(self, src_tokens, src_lengths, prev_output_tokens, return_all_hiddens: bool = True,
                features_only: bool = False, alignment_layer: Optional[int] = None,
                alignment_heads: Optional[int] = None, grid_img_features=None,
                region_img_features=None):

        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths,
                                   return_all_hiddens=return_all_hiddens, grid_img_features=grid_img_features,
                                   region_img_features=region_img_features)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out,
                                   features_only=features_only, alignment_layer=alignment_layer,
                                   alignment_heads=alignment_heads, src_lengths=src_lengths,
                                   return_all_hiddens=return_all_hiddens)

        return decoder_out


@register_model_architecture('rgmmt_transformer', 'rgmmt_model')
def my_hyperparameters(args):
    base_architecture(args)


