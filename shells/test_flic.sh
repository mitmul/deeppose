#!/bin/bash

python scripts/evaluate_flic.py \
--model results/ResNet50_2016-06-30_00-22-18/AlexNet.py \
--param results/ResNet50_2016-06-30_00-22-18/epoch-1.model \
--gpu 1 \
--symmetric_joints "[[2, 4], [1, 5], [0, 6]]"
