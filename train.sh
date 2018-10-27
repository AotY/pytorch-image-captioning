#!/usr/bin/env bash
#
# train.sh
# Copyright (C) 2018 LeonTao 
#
# Distributed under terms of the MIT license.
#


export CUDA_VISIBLE_DEVICES=5

python train.py \
    --model_path ./models \
    --crop_size 224 \
    --vocab_path ./data/vocab.pkl \
    --data_dir ./data/resized2014 \
    --caption_path ./data/annotations/captions_train2014.json \
    --log_interval 20 \
    --save_step 1000 \
    --max_len 25 \
    --embedding_size 256 \
    --hidden_size 512 \
    --num_layers 1 \
    --device cuda \
    --epochs 5 \
    --batch_size 128 \
    --num_workers 2 \
    --lr 0.001  \

/
