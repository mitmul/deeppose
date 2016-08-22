#!/bin/bash

CHAINER_TYPE_CHECK=0 \
python scripts/train.py \
--model models/AlexNet.py \
--gpus 8 \
--epoch 100 \
--batchsize 128 \
--snapshot 10 \
--train_csv_fn data/FLIC-full/train_joints.csv \
--test_csv_fn data/FLIC-full/test_joints.csv \
--img_dir data/FLIC-full/images \
--test_freq 10 \
--seed 1701 \
--im_size 220 \
--fliplr \
--rotate \
--rotate_range 10 \
--zoom \
--zoom_range 0.2 \
--translate \
--translate_range 5 \
--coord_normalize \
--gcn \
--n_joints 7 \
--fname_index 0 \
--joint_index 1 \
--symmetric_joints "[[2, 4], [1, 5], [0, 6]]" \
--opt Adam
