#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
image caption
1. dataset
2. model
   encoder
   decoder
3. criterion
4. optimizer
4. loop
"""
import os
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from attn_seq2seq import AttnSeq2seq
from coco_dataset import CocoDataset
from vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    default='models/', help='path for saving trained models')
parser.add_argument('--crop_size', type=int, default=224,
                    help='size for randomly cropping images')
parser.add_argument('--vocab_path', type=str,
                    default='data/vocab.pkl', help='path for vocabulary wrapper')
parser.add_argument('--data_dir', type=str,
                    default='data/resized2014', help='directory for resized images')
parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json',
                    help='path for train annotation json file')
parser.add_argument('--log_interval', type=int, default=10,
                    help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=1000,
                    help='step size for saving trained models')
parser.add_argument('--max_len', type=int, default=20,
                    help='max len of a caption.')

# Model parameters
parser.add_argument('--encode_image_size', type=int, default=14,
                    help='image size after decoding.')
parser.add_argument('--encoder_size', type=int, default=2048,
                    help='encoder size, default 2048 (from resnet152)')
parser.add_argument('--embedding_size', type=int, default=256,
                    help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int, default=512,
                    help='dimension of lstm hidden states')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in lstm')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout probability.')

parser.add_argument('--device', type=str, default='cuda',
                    help='cuda or cpu, can specifying a number')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001)

args = parser.parse_args()
device = torch.device(args.device)


def load_vocab():
    vocab = Vocab()
    vocab.load(args.vocab_path)
    return vocab


def build_data_loader(vocab):
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    coco_dataset = CocoDataset(
        data_dir=args.data_dir,
        caption_path=args.caption_path,
        vocab=vocab,
        max_len=args.max_len,
        transform=transform,
        device=device)

    data_loader = coco_dataset.get_loader(args.batch_size,
                                      shuffle=True,
                                      num_workers=2)

    return data_loader


def build_model(vocab_size, padid):
    model = AttnSeq2seq(
        args.encode_image_size,
        vocab_size,
        args.encoder_size,
        args.embedding_size,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        padid,
        args.max_len
    ).to(device)
    print(model)
    return model


def build_criterion(padid):
    criterion = nn.NLLLoss(ignore_index=padid)
    return criterion


def build_optimizer(model):
    optimizer = optim.Adam(model.parameters(), args.lr)
    return optimizer


def train_epoch(model, data_loader, criterion, optimizer):
    total_step = len(data_loader)
    for epoch in range(1, 1 + args.epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            print('images shape: {}'.format(images.shape))
            print('captions shape: {}'.format(captions.shape))

            images = images.to(device)
            captions = captions.to(device)

            #  targets, _ = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True)

            outputs = model(images, captions, lengths)

            optimizer.zero_grad()

            # loss
            loss = criterion(outputs.view(-1, decoder.vocab_size), captions.view(-1))

            loss.backward()
            optimizer.step()

            # Print log info
            if (i + 1) % args.log_interval == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}' \
                      .format(epoch, args.epochs, i, total_step, loss.item(), np.exp(loss.item())))
            if (i + 1) % args.save_step == 0:
                    torch.save(decoder.state_dict(), os.path.join(
                        args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                    torch.save(encoder.state_dict(), os.path.join(
                        args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))


if __name__ == '__main__':
    vocab = load_vocab()
    data_loader = build_data_loader(vocab)
    print(data_loader)
    model = build_model(vocab.size, vocab.padid)
    criterion = build_criterion(vocab.padid)
    optimizer = build_optimizer(model)
    train_epoch(
        model,
        data_loader,
        criterion,
        optimizer
    )
