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
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--snapshot', type=int, default=10)
    parser.add_argument('--channel', type=int, default=3)
    parser.add_argument('--test_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--ignore_label', type=float, default=-1)
    parser.add_argument(
        '--model', type=str, default='models/AlexNet_flic.py',
        help='Model definition file in models dir')
    parser.add_argument(
        '--gpus', type=str, default='0',
        help='Comma-separated GPU IDs (e.g., "0,1,2")')
    parser.add_argument(
        '--train_csv_fn', type=str,
        default='data/FLIC-full/train_joints.csv')
    parser.add_argument(
        '--test_csv_fn', type=str,
        default='data/FLIC-full/test_joints.csv')
    parser.add_argument(
        '--img_dir', type=str,
        default='data/FLIC-full/images')
    parser.add_argument(
        '--valid_freq', type=int, default=5,
        help='Perform test every this epoch (0 means no test)')
    parser.add_argument(
        '--show_log_iter', type=int, default=10,
        help='Show loss value per this iterations')

    # Data argumentation settings
    parser.add_argument(
        '--im_size', type=int, default=220,
        help='Resize input image into this big')
    parser.add_argument(
        '--fliplr', action='store_true', default=False,
        help=('Flip image\'s left and right for data augmentation'))
    parser.add_argument(
        '--rotate', action='store_true', default=False,
        help=('Randomly rotate images for data augmentation'))
    parser.add_argument(
        '--rotate_range', type=int, default=10,
        help=('The max angle(degree) of rotation for data augmentation'))
    parser.add_argument(
        '--zoom', action='store_true', default=False,
        help=('Randomly zoom out/in images for data augmentation'))
    parser.add_argument(
        '--base_zoom', type=float, default=1.5,
        help=('How big is the input image region comapred to bbox of joints'))
    parser.add_argument(
        '--zoom_range', type=float, default=0.2,
        help=('The max zooming amount for data augmentation'))
    parser.add_argument(
        '--translate', action='store_true', default=False,
        help=('Randomly translate images for data augmentation'))
    parser.add_argument(
        '--translate_range', type=int, default=5,
        help=('The max size of random translation for data augmentation'))
    parser.add_argument(
        '--min_dim', type=int, default=0,
        help='Minimum dimension of a person')
    parser.add_argument(
        '--coord_normalize', action='store_true', default=False,
        help=('Perform normalization to all joint coordinates'))
    parser.add_argument(
        '--gcn', action='store_true', default=False,
        help=('Perform global contrast normalization for each input image'))

    # Data configuration
    parser.add_argument('--n_joints', type=int, default=7)
    parser.add_argument(
        '--fname_index', type=int, default=0,
        help='the index of image file name in a csv line')
    parser.add_argument(
        '--joint_index', type=int, default=1,
        help='the start index of joint values in a csv line')
    parser.add_argument(
        '--symmetric_joints', type=str, default='[[2, 4], [1, 5], [0, 6]]',
        help='Symmetric joint ids in JSON format')
    # flic_swap_joints = [(2, 4), (1, 5), (0, 6)]
    # lsp_swap_joints = [(8, 9), (7, 10), (6, 11), (2, 3), (1, 4), (0, 5)]
    # mpii_swap_joints = [(12, 13), (11, 14), (10, 15), (2, 3), (1, 4), (0, 5)]

    # Optimization settings
    parser.add_argument(
        '--opt', type=str, default='Adam',
        choices=['MomentumSGD', 'Adam', 'AdaGrad', 'RMSprop'],
        help='Optimization method')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--adam_alpha', type=float, default=0.001)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument(
        '--lr_decay_freq', type=int, default=10,
        help='The learning rate will be decreased every this epoch')
    parser.add_argument(
        '--lr_decay_ratio', type=float, default=0.1,
        help='When the learning rate is decreased, this number will be'
             'multiplied')

    # Resuming
    parser.add_argument(
        '--resume_model', type=str, default=None,
        help='Load model definition file to use for resuming training')
    parser.add_argument(
        '--resume_param', type=str, default=None,
        help='Load learnt model parameters from this file (it\'s necessary'
             'when you resume a training)')
    parser.add_argument(
        '--resume_opt', type=str, default=None,
        help='Load optimization states from this file (it\'s necessary'
             'when you resume a training)')

    args = parser.parse_args()
    args.epoch += 1

    return args
