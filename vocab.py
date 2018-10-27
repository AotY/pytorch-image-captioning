#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

# -*- coding: utf-8 -*-
"""
    Manage Vocabulary,
        1), Build Vocabulary
        2), save and load vocabulary
"""
from __future__ import division
from __future__ import print_function
import nltk
import pickle
import logging
import argparse
from pycocotools.coco import COCO

logger = logging.getLogger(__name__)

PAD = '<pad>'
UNK = '<unk>'
SOS = '<sos>'
EOS = '<eos>'


class Vocab(object):
    def __init__(self):
        self.init_vocab()

    def init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = {}

        self.word_count[PAD] = 50000000
        self.word_count[UNK] = 40000000
        self.word_count[SOS] = 30000000
        self.word_count[EOS] = 20000000

    '''word to id '''

    def word_to_id(self, word):
        return self.word2idx.get(word, self.unkid)

    def words_to_id(self, words):
        word_ids = [self.word_to_id(cur_word) for cur_word in words]
        word_ids = [id for id in word_ids if id != self.unkid]
        return word_ids

    '''id to  word'''

    def id_to_word(self, id):
        return self.idx2word.get(id, self.unk)

    '''ids to word'''
    def ids_to_word(self, ids):
        words = [self.id_to_word(id) for id in ids]
        words = [word for word in words if word != self.unk]
        return words

    def add_words(self, words):
        for word in words:
            self.word_count[word] = self.word_count.get(word, 0) + 1

    def build_vocab(self, min_count, vocab_size=None):
        freq_list = sorted(self.word_count.items(), key=lambda item: item[1], reverse=True)
        freq_list = [item for item in freq_list if item[1] > min_count]
        # clip by vocab_size
        if vocab_size:
            freq_list = freq_list[:vocab_size]
        for i, (word, _) in enumerate(freq_list):
            self.word2idx[word] = i
            self.idx2word[i] = word

    '''save and restore'''

    def save(self, path=None):
        if len(self.idx2word) == 0:
            raise RuntimeError("Save vocab after call build_from_freq()")

        pickle.dump(self.word2idx, open(path, 'wb'))

    def load(self, path=None):
        try:
            self.word2idx = pickle.load(open(path, 'rb'))
            self.idx2word = {v: k for k, v in self.word2idx.items()}
        except:
            raise RuntimeError("Make sure vocab_word2idx.dict exists.")

    @property
    def size(self):
        return len(self.word2idx)

    ''' wordid '''

    @property
    def padid(self):
        """return the id of padding
        """
        return self.word2idx.get(PAD, 0)


    @property
    def unkid(self):
        """return the id of unknown word
        """
        return self.word2idx.get(UNK, 1)

    @property
    def sosid(self):
        """return the id of padding
        """
        return self.word2idx.get(SOS, 2)

    @property
    def eosid(self):
        """return the id of padding
        """
        return self.word2idx.get(EOS, 3)

    '''words '''

    @property
    def unk(self):
        """return the str of unknown word
        """
        return UNK

    @property
    def pad(self):
        """return the str of padding
        """
        return PAD

    @property
    def sos(self):
        """return the str of padding
        """
        return SOS

    @property
    def eos(self):
        """return the str of padding
        """
        return EOS

    def ids_to_text(self, ids):
        words = self.ids_to_word(ids)
        # remove pad, sos, eos, unk
        words = [word for word in words if word not in [self.pad, self.unk, self.sos, self.eos]]
        text = ' '.join(words)
        return text



def build_vocab(args):
    vocab = Vocab()
    """Build a simple vocabulary wrapper."""
    coco = COCO(args.caption_path)
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        vocab.add_words(tokens)
    vocab.build_vocab(args.min_count)

    return vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default='data/annotations/captions_train2014.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--min_count', type=int, default=4,
                        help='minimum word count min_count')
    #  parser.add_argument('--vocab_size', type=int, default=30000,
                        #  help='vocab size.')
    args = parser.parse_args()

    vocab = build_vocab(args)
    print('vocab size : %d' % vocab.size)
    vocab.save(args.vocab_path)
