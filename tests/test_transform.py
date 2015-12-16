#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import cv2 as cv
import argparse
import sys
sys.path.append('scripts')
from transform import Transform
from test_flic_dataset import draw_joints


def load_data(trans, args, x):
    img_fn = '%s/images/%s' % (args.data_dir, x.split(',')[args.fname_index])
    print(os.path.exists(img_fn), img_fn)
    orig = cv.imread(img_fn)

    c = args.channel
    s = args.size
    d = args.joint_num * 2

    # data augmentation
    input_data = np.zeros((1, c, s, s))
    label = np.zeros((1, d))
    x, t = trans.transform(x.split(','), args.data_dir)
    input_data[0] = x.transpose((2, 0, 1))
    label[0] = t

    return orig, input_data, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, default='data/FLIC-full')
    parser.add_argument('--channel', '-c', type=int, default=3)
    parser.add_argument('--flip', type=bool, default=True,
                        help='flip left and right for data augmentation')
    parser.add_argument('--size', type=int, default=220,
                        help='resizing')
    parser.add_argument('--crop_pad_inf', type=float, default=1.5,
                        help='random number infimum for padding size when'
                             'cropping')
    parser.add_argument('--crop_pad_sup', type=float, default=2.0,
                        help='random number supremum for padding size when'
                             'cropping')
    parser.add_argument('--shift', type=int, default=5,
                        help='slide an image when cropping')
    parser.add_argument('--lcn', type=bool, default=True,
                        help='local contrast normalization for data'
                             'augmentation')
    parser.add_argument('--joint_num', type=int, default=7)
    parser.add_argument('--fname_index', type=int, default=0,
                        help='the index of image file name in a csv line')
    args = parser.parse_args()
    print(args)

    # augmentation setting
    trans = Transform(args)

    # test data
    test_fn = '%s/test_joints.csv' % args.data_dir
    test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

    result_dir = '%s/test_trans' % args.data_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for i, line in enumerate(test_dl):
        orig, input_data, label = load_data(trans, args, line)
        input_data = input_data.transpose((0, 2, 3, 1))[0].astype(np.float32)
        label = label.astype(np.float32).flatten()
        cv.imwrite('%s/%d_orig.jpg' % (result_dir, i), orig)
        img, label = trans.revert(input_data, label)
        label = [tuple(l) for l in label]
        pose = draw_joints(input_data.copy(), label)
        pose = np.array(pose.copy())
        cv.imwrite('%s/%d_pose.jpg' % (result_dir, i), pose)
