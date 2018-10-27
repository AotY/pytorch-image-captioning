#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""
import os
import nltk
from PIL import Image

import torch
import torch.utils.data as data

from pycocotools.coco import COCO

class CocoDataset(data.Dataset):
    def __init__(self,
                 data_dir,
                 caption_path,
                 vocab,
                 max_len,
                 transform=None,
                 device=None):
        self.data_dir = data_dir
        self.coco = COCO(caption_path)
        self.vocab = vocab
        self.transform = transform
        self.device = device

        self.ann_ids = list(self.coco.anns.keys())

    def __len__(self):
        return len(self.ann_ids)

    def __getitem__(self, index):
        """ return a pair (image, caption) """
        ann_id = self.ann_ids[index]
        caption = self.coco.anns[ann_id]['caption']

        image_id = self.coco.anns[ann_id]['image_id']
        image_path = os.path.join(self.data_dir, self.coco.loadImgs(image_id)[0]['file_name'])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # caption to caption ids
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption_ids = []
        # here, because the image feature is the first input of lstm.
        caption_ids.append(self.vocab.sosid)
        caption_ids.extend([self.vocab.word_to_id(token) for token in tokens])
        caption_ids.append(self.vocab.eosid)

        caption = torch.tensor(caption_ids, dtype=torch.long, device=self.device)
        return image, caption


    def collate_fn(self, pairs):
        """Create mini-batch tensors for the pairs (image, caption).
        pairs:
            - image: [3, 256, 256]
            - caption: [len]
        """
        pairs.sort(key=lambda item: len(item[1]), reverse=True)
        images, captions = zip(*pairs)

        # merge images (from list of 3D tensor to 4D tensor)
        images = torch.stack(images, dim=0)

        # merge captions (from 1D tensor to 2D tensor)
        lengths = [len(caption) for caption in caption_ids]

        padded_captions = torch.ones((len(captions), self.max_len), dtype=torch.long, device=self.device) * self.vocab.padid
        for i, caption in enumerate(captions):
            padded_captions[i, :lengths[i]] = caption[:lengths[i]]

        return images, padded_captions, lengths

    def get_loader(self, batch_size, shuffle=True, num_workers=1):
        data_loader = data.DataLoader(dataset=self,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=num_workers,
                                      collate_fn=self.collate_fn)

        return data_loader
