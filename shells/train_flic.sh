#!/bin/bash

CHAINER_TYPE_CHECK=0 \
python scripts/train.py \
--model models/AlexNet.py \
--gpu 2 \
--epoch 1000 \
--batchsize 128 \
--snapshot 10 \
--datadir data/FLIC-full \
--channel 3 \
--test_freq 10 \
--seed 1701 \
--flip 1 \
--size 220 \
--min_dim 100 \
--cropping 1 \
--crop_pad_inf 1.4 \
--crop_pad_sup 1.6 \
--shift 5 \
--gcn 1 \
--joint_num 7 \
--fname_index 0 \
--joint_index 1 \
--symmetric_joints "[[2, 4], [1, 5], [0, 6]]" \
--opt Adam
