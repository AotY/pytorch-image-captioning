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
using attention mechanism

"""

class AttnEncoder(nn.Module):
    def __init__(self, encode_image_size):
        super(AttnEncoder, self).__init__()

        resnet152 = torchvision.models.resnet152(pretrained=True)
        modules = list[resnet152.children()][:-2] # remove avgpool and fc layer.

        self.cnn = nn.Sequential(*modules)

        # resize image to fixed size to allow input images of variable size.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encode_image_size, encode_image_size))

    def forward(self, images):
        """
        Args:
            images: [batch_size, 3, 256, 256]
        return:
            outputs: [batch_size, encode_image_size, encode_image_size, 2048]

        """
        print('images shape: {}'.format(images.shape))
        # Extract feature from images
        with torch.no_grad():
            features = self.cnn(images) #[batch_size, 2048, w/32, w/32]
            print('features shape: {}'.format(features.shape))

        outputs = self.adaptive_pool(features) #[batch_size, 2048, encode_image_size, encode_image_size]
        outputs = outputs.permute(0, 2, 3, 1) #[batch_size, encode_image_size, encode_image_size, 2048]
        print('outputs shape: {}'.format(outputs.shape))

        return outputs


