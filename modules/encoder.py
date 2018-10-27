#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
CNN encoder.
Using resnet-152 model pretrained on the ILSVRC-2012-CLS image classification dataset.
"""

import torch
import torch.nn as nn
import torch.functional as F
import torchvision

"""
without attention mechanism

using fc layer features.
"""

class Encoder(nn.Module):
    def __init__(self, embedding_size):
        super(Encoder, self).__init__()

        resnet152 = torchvision.models.resnet152(pretrained=True)
        modules = list(resnet152.children())[:-1] # remove fc layer

        self.cnn = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet152.fc.in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size, momentum=0.01)

    def forward(self, images):
        print('images: {}'.format(images.shape))

        """ Extract feature from images"""
        with torch.no_grad():
            features = self.cnn(images)

        features = features.reshape(features.shape[0], -1) # squeeze
        outputs = self.bn(self.linear(features))
        print('encoder outpus shape: {}'.format(outputs.shape))

        return outputs







