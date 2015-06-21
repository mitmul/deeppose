#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import cv2 as cv
from chainer import cuda
import imp
import argparse
import sys
sys.path.append('scripts')
from transform import Transform
from test_flic_dataset import draw_joints


def load_data(trans, args, x):
    c = args.channel
    s = args.size
    d = args.joint_num * 2

    # data augmentation
    input_data = np.zeros((1, c, s, s))
    label = np.zeros((1, d))
    x, t = trans.transform(x.split(','), args.data_dir)
    input_data[0] = x.transpose((2, 0, 1))
    label[0] = t

    return trans.orig, input_data, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', '-m', type=str)
    parser.add_argument('--data_dir', '-d', type=str, default='data/FLIC-full')
    parser.add_argument('--channel', '-c', type=int, default=3)
    parser.add_argument('--size', '-s', type=int, default=227)
    parser.add_argument('--shift', '-f', type=int, default=5)
    parser.add_argument('--joint_num', '-j', type=int, default=7)
    args = parser.parse_args()
    print(args)

    # augmentation setting
    trans = Transform(padding=[1.5, 2.0],
                      flip=True,
                      size=args.size,
                      shift=5,
                      norm=True)

    # test data
    test_fn = '%s/test_joints.csv' % args.data_dir
    test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

    result_dir = '%s/test_trans' % args.data_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for i, line in enumerate(test_dl):
        orig, input_data, label = load_data(trans, args, line)
        input_data = input_data.transpose((0, 2, 3, 1))[0]
        cv.imwrite('%s/%d_orig.jpg' % (result_dir, i), orig)
        cv.imwrite('%s/%d_trans.jpg' % (result_dir, i), input_data)

        label = np.array(trans.revert(label), dtype=np.int32)[0]
        label = zip(label[0::2], label[1::2])
        pose = draw_joints(input_data.copy(), label)
        pose = np.array(pose.copy())
        cv.imwrite('%s/%d_pose.jpg' % (result_dir, i), pose)
