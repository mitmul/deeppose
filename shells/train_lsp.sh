#!/bin/bash

CHAINER_TYPE_CHECK=0 \
python scripts/train.py \
--model models/AlexNet.py \
--gpu 0 \
--epoch 1000 \
--batchsize 16 \
--snapshot 10 \
--datadir data/lspet_dataset \
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
--joint_num 14 \
--fname_index 0 \
--joint_index 1 \
--symmetric_joints "[(8, 9), (7, 10), (6, 11), (2, 3), (1, 4), (0, 5)]" \
--opt Adam
