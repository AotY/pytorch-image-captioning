#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
RNN Decoder with attention.

Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self,
                 encoder_size,
                 decoder_size,
                 atten_size):
        super(Attention, self).__init__()
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.atten_size = atten_size

        self.encoder_linear = nn.Linear(encoder_size, atten_size)
        self.decoder_linear = nn.Linear(decoder_size, atten_size)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        Args:
            - encoder_outputs: [batch_size, num_pixels, encoder_size],
            num_pixels = encode_image_size * encode_image_size
            - decoder_hidden: [batch_size, 1, decoder_size]

        """
        attn_vectors = self.encoder_linear(encoder_outputs)
        decoder_vector = self.decoder_linear(decoder_hidden)

        attn_weights = torch.bmm(decoder_vector, attn_vectors.transpose(1, 2)) #[batch_size, 1, num_pixels]
        attn_weights = F.softmax(attn_weights, dim=2)
        attn_vector = torch.bmm(attn_weights, attn_vectors)

        return attn_vector, attn_weights


"""
attention with
Decoder.
"""
class AttnDecoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 encoder_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 padding_idx,
                 max_len):

        super(AttnDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size,
                                      embedding_size,
                                      padding_idx=padding_idx)

        self.attn = Attention(
            encoder_size=encoder_size,
            decoder_size=embedding_size,
            atten_size=hidden_size
        )

        self.lstm = nn.LSTM(embedding_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)

        self.linear = nn.Linear(hidden_size, vocab_size)

        self.log_softmax = nn.LogSoftmax(dim=2)


    def forward(self, captions, lengths, encoder_outputs):
        """
        Args:
            captions: [batch_size, len]
            lengths: [batch_size]
            encoder_outputs: [batch_size, num_pixels, encoder_size]
        """

        embedded = self.embedding(captions) # [batch_size, len, embedding_size]
        embedded = self.dropout(embedded)


        context_embedded, attn_weights = self.attn(encoder_outputs, embedded)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequeence(context_embedded, lengths, batch_first=True)
            outputs, hidden_state = self.lstm(packed)
            print('outputs shape: {}'.format(outputs.shape))
        else:
            outputs, hidden_state = self.lstm(context_embedded)

        outputs = self.linear(outputs)

        outputs = self.log_softmax(outputs)

        return outputs, attn_weights



