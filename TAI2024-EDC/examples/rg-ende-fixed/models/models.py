# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple
from fairseq.models import register_model, register_model_architecture
from .encoder import RGFIXEncoder
from .decoder import RGFIXDecoder
from fairseq.models.transformer import (TransformerModel, base_architecture, TransformerDecoder)


@register_model("rgfix_transformer")
class RGMMTModel(TransformerModel):
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

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @classmethod
    def build_model(cls, args, task):
        model = super().build_model(args, task)

        # if args.training_stage==0 or args.training_stage==2:
        #     for name, p in model.named_parameters():
        #         if 'attacker' in name:
        #             p.requires_grad = False
        #         else:
        #             p.requires_grad = True
        # else:
        #     for name, p in model.named_parameters():
        #         if 'attacker' not in name:
        #             p.requires_grad = False
        #         else:
        #             p.requires_grad = True

        return model

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return RGFIXEncoder(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return RGFIXDecoder(args, tgt_dict, embed_tokens,
                            no_encoder_attn=getattr(args, "no_cross_attention", False))

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
                                   region_img_features=region_img_features, loss_type='nmt')

        encoder_out_mask = self.encoder(src_tokens, src_lengths=src_lengths,
                                        return_all_hiddens=return_all_hiddens, grid_img_features=grid_img_features,
                                        region_img_features=region_img_features, loss_type='mask')

        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, features_only=features_only,
                                   alignment_layer=alignment_layer, alignment_heads=alignment_heads,
                                   src_lengths=src_lengths, return_all_hiddens=return_all_hiddens,
                                   loss_type='nmt')

        decoder_out_mask = self.decoder(prev_output_tokens, encoder_out=encoder_out_mask, features_only=features_only,
                                        alignment_layer=alignment_layer, alignment_heads=alignment_heads,
                                        src_lengths=src_lengths, return_all_hiddens=return_all_hiddens,
                                        loss_type='mask')

        return decoder_out, decoder_out_mask

    # def set_stage(self, stage=0):
    #     self.decoder.set_stage(stage)
    # print('stage %d: is_attack %s att_replace %s'%
    #     (stage, self.decoder.is_attack, self.decoder.att_replace))


@register_model_architecture('rgfix_transformer', 'rgfix_model')
def my_hyperparameters(args):
    base_architecture(args)