#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument(
        '--model', type=str, default='models/AlexNet_flic.py',
        help='model definition file in models dir')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--snapshot', type=int, default=10)
    parser.add_argument('--datadir', type=str, default='data/FLIC-full')
    parser.add_argument('--channel', type=int, default=3)

    # Data argumentation settings
    parser.add_argument(
        '--flip', type=int, default=1,
        help='flip left and right for data augmentation')
    parser.add_argument('--size', type=int, default=220, help='resizing')
    parser.add_argument(
        '--crop_pad_inf', type=float, default=1.5,
        help='random number infimum for padding size when cropping')
    parser.add_argument(
        '--crop_pad_sup', type=float, default=2.0,
        help='random number supremum for padding size when cropping')
    parser.add_argument(
        '--shift', type=int, default=5, help='slide an image when cropping')
    parser.add_argument(
        '--lcn', type=int, default=1,
        help='local contrast normalization for data augmentation')

    # Data configuration
    parser.add_argument('--joint_num', type=int, default=7)
    parser.add_argument('--fname_index', type=int, default=0,
                        help='the index of image file name in a csv line')
    parser.add_argument('--joint_index', type=int, default=1,
                        help='the start index of joint values in a csv line')

    # Optimization settings
    parser.add_argument('--resume_model', type=str, default=None,
                        help='*.model file path to resume from')
    parser.add_argument('--resume_opt', type=str, default=None,
                        help='*.state file path to resume from')
    parser.add_argument(
        '--epoch_offset', type=int, default=0,
        help='set greater than 0 if you restart from a saved model')
    parser.add_argument('--opt', type=str, default='AdaGrad',
                        choices=['AdaGrad', 'MomentumSGD', 'Adam'])
    args = parser.parse_args()

    return args
