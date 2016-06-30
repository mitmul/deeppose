#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2 as cv
import numpy as np
import os
import sys


def load_data(trans, args, x):
    img_fn = '%s/images/%s' % (args.datadir, x.split(',')[args.fname_index])
    print(os.path.exists(img_fn), img_fn)
    orig = cv.imread(img_fn)

    c = args.channel
    s = args.size
    d = args.joint_num * 2

    # data augmentation
    input_data = np.zeros((1, c, s, s))
    label = np.zeros((1, d))
    x, t = trans.transform(x.split(','), args.datadir)
    input_data[0] = x.transpose((2, 0, 1))
    label[0] = t

    return orig, input_data, label


if __name__ == '__main__':
    sys.path.append('scripts')

    from test_flic_dataset import draw_joints as flic_draw_joints
    from test_lsp_dataset import draw_joints as lsp_draw_joints
    from test_mpii_dataset import draw_joints as mpii_draw_joints
    from transform import Transform
    from cmd_options import get_arguments

    args = get_arguments()
    print(args)

    flic_swap_joints = [(2, 4), (1, 5), (0, 6)]
    lsp_swap_joints = [(8, 9), (7, 10), (6, 11), (2, 3), (1, 4), (0, 5)]
    mpii_swap_joints = [(12, 13), (11, 14), (10, 15), (2, 3), (1, 4), (0, 5)]

    for datadir, n_joints, draw_joints, swap_joints, min_dim in [
            ('data/FLIC-full', 7, flic_draw_joints, flic_swap_joints, 100),
            ('data/lspet_dataset', 14, lsp_draw_joints, lsp_swap_joints, 50),
            ('data/mpii', 16, mpii_draw_joints, mpii_swap_joints, 100)]:
        args.datadir = datadir
        args.joint_num = n_joints
        args.min_dim = min_dim
        # augmentation setting
        trans = Transform(args, swap_joints)

        # test data
        test_fn = '%s/train_joints.csv' % args.datadir
        test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

        result_dir = '%s/test_trans' % args.datadir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for i, line in enumerate(test_dl):
            orig, input_data, label = load_data(trans, args, line)
            input_data = input_data.transpose(
                (0, 2, 3, 1))[0].astype(np.float32)
            label = label.astype(np.float32).flatten()
            cv.imwrite('%s/%d_orig.jpg' % (result_dir, i), orig)
            img, label = trans.revert(input_data, label)
            label = [tuple(l) for l in label]
            pose = draw_joints(input_data.copy(), label)
            cv.rectangle(pose, trans.bbox[0], trans.bbox[1], (0, 0, 255))
            pose = np.array(pose.copy())
            cv.imwrite('%s/%d_pose.jpg' % (result_dir, i), pose)
            if i > 100:
                break
