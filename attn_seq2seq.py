#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.attn_encoder import AttnEncoder
from modules.attn_decoder import AttnDecoder


class AttnSeq2seq(nn.Module):
    def __init__(self,
                 encode_image_size,
                 vocab_size,
                 encoder_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 padding_idx,
                 max_len):
        super(AttnSeq2seq, self).__init__()
        self.max_len = max_len

        self.encoder = AttnEncoder(
            encode_image_size=encode_image_size
        )

        self.decoder = AttnDecoder(
            vocab_size,
            encoder_size,
            embedding_size
            hidden_size,
            num_layers,
            dropout,
            padding_idx,
            max_len
        )

    def forward(self, images, captions, lengths):
        """
        Args:
            images: [batch_size, 256, 256]
            captions: [batch_size, max_len]
            lengths: [batch_size]

        """
        encoder_outputs = self.encoder(
            images)  # [batch_size, encode_image_size, encode_image_size, 2048]

        decoder_outputs, attn_weights = self.decoder(
            captions,
            lengths,
            encoder_outputs
        )

        return decoder_outputs

    def decode(self, images, decoder_input):
        """
        Args:
            images: [batch_size, 256, 256]
            decoder_input: [batch_size, 1]
        """
        batch_size, _ = decoder_input
        encoder_outputs = self.encoder(images)
        decode_outputs = []
        for i in range(self.max_len):
            decoder_output, attn_weight = self.decoder.decode(decoder_input,
                                                       encoder_outputs)
            # decoder_output: [batch_size, 1, vocab_size]
            decoder_input = decoder_output.argmax(dim=2)
            decode_outputs.append(decoder_input)

        decode_outputs = torch.cat(decode_outputs, dim=1)

        return decode_outputs

