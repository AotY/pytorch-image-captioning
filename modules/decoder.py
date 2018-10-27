#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
RNN Decoder

Show ant Tell

images feature only using one time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 padding_idx,
                 max_len):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size,
                                      embedding_size,
                                      padding_idx=padding_idx)

        self.lstm = nn.LSTM(embedding_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)

        self.linear = nn.Linear(hidden_size, vocab_size)

        self.log_softmax = nn.LogSoftmax(dim=2)


    def forward(self, captions, lengths, images_feature):
        """
        Args:
            captions: [batch_size, len]
            lengths: [batch_size]
            images_feature: [batch_size, embedding_size]
        """

        embedded = self.embedding(captions) # [batch_size, len, embedding_size]
        print('embedded shape: {}'.format(embedded.shape))
        embedded = self.dropout(embedded)
        embedded_cat = torch.cat((images_feature.unsqueeze(1), embedded), dim=1)
        print('embedded_cat shape: {}'.format(embedded_cat.shape))

        packed = nn.utils.rnn.pack_padded_sequence(embedded_cat, lengths, batch_first=True)
        packed_outputs, hidden_state = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs,
                                                      batch_first=True,
                                                      total_length=captions.size(1))
        print('outputs shape: {}'.format(outputs.shape))

        outputs = self.linear(outputs)

        outputs = self.log_softmax(outputs)
        print('decoder outputs shape: {}'.format(outputs.shape))

        return outputs


    def sample(self, images_feature, states=None):
        """Generate captions for given images using greedy search."""
        sampled_ids = []
        input = images_feature.unsqueeze(1) #[batch_size, 1, embedding_size]
        for i in range(self.max_len):
            output, state = self.lstm(input, states)
            output = self.linear(output) # [batch_size, 1, vocab_size]
            output = self.log_softmax(output) #[batch_size, 1, vocab_size]
            output.squeeze(1) # [batch_size, vocab_size]
            _, predicted = output.max(dim=1) #[batch_size]
            sampled_ids.append(predicted)
            input = self.embedding(predicted).unsqueeze(1)

        sampled_ids = torch.stack(sampled_ids, dim=1) #[batch_size, max_len]
        return sampled_ids






