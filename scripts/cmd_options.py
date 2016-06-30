#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument(
        '--model', type=str, default='models/AlexNet_flic.py',
        help='Model definition file in models dir')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--snapshot', type=int, default=10)
    parser.add_argument('--datadir', type=str, default='data/FLIC-full')
    parser.add_argument('--channel', type=int, default=3)
    parser.add_argument('--test_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1701)

    # Data argumentation settings
    parser.add_argument(
        '--flip', type=int, default=1,
        help=('Flip left and right as data augmentation. '
              '--flip 1 means it performs LR flip augmentation, '
              'while --flip 0 does nothing.'))
    parser.add_argument('--size', type=int, default=220,
                        help='Resize input image into this big.')
    parser.add_argument('--min_dim', type=int, default=100,
                        help='Minimum dimension of a person.')
    parser.add_argument('--cropping', type=int, default=1)
    parser.add_argument(
        '--crop_pad_inf', type=float, default=1.4,
        help=('Minimum value for random padding size during data augmentation'
              ' by cropping input image'))
    parser.add_argument(
        '--crop_pad_sup', type=float, default=1.6,
        help=('Maximum value for random padding size during data augmentation'
              ' by cropping input image'))
    parser.add_argument(
        '--shift', type=int, default=5,
        help=('Maximum value for random translation size during data'
              ' augmentation by translating input image'))
    parser.add_argument(
        '--gcn', type=int, default=1,
        help=('Perform (1) or not (0) global contrast normalization after'
              ' data augmentation process to input image'))

    # Data configuration
    parser.add_argument('--joint_num', type=int, default=7)
    parser.add_argument('--fname_index', type=int, default=0,
                        help='the index of image file name in a csv line')
    parser.add_argument('--joint_index', type=int, default=1,
                        help='the start index of joint values in a csv line')
    parser.add_argument(
        '--symmetric_joints', type=str, default='[[2, 4], [1, 5], [0, 6]]',
        help='Symmetric joint ids in JSON format')
    # flic_swap_joints = [(2, 4), (1, 5), (0, 6)]
    # lsp_swap_joints = [(8, 9), (7, 10), (6, 11), (2, 3), (1, 4), (0, 5)]
    # mpii_swap_joints = [(12, 13), (11, 14), (10, 15), (2, 3), (1, 4), (0, 5)]

    # Optimization settings
    parser.add_argument('--resume_model', type=str, default=None,
                        help='*.model file path to resume from')
    parser.add_argument('--resume_opt', type=str, default=None,
                        help='*.state file path to resume from')
    parser.add_argument('--epoch_offset', type=int, default=0,
                        help='set greater than 0 if you restart from a saved'
                             ' model')
    parser.add_argument('--opt', type=str, default='AdaGrad',
                        choices=['AdaGrad', 'MomentumSGD', 'Adam'])
    parser.add_argument('--lr', type=float, default=0.0005)
    args = parser.parse_args()

    return args
