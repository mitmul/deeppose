#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import re
import argparse
import os
import numpy as np
import cv2 as cv
from scipy.io import loadmat
import glob
from os.path import basename as b

if __name__ == '__main__':
    # to fix test set
    np.random.seed(1701)

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data/lspet_dataset')
    args = parser.parse_args()
    print(args)

    jnt_fn = '%s/joints.mat' % args.datadir
    joints = loadmat(jnt_fn)
    joints = joints['joints'].swapaxes(0, 2).swapaxes(1, 2)
    joints = np.abs(joints[:, :, :2])

    perm = np.random.permutation(int(len(joints) * 0.1))
    print(perm)

    fp = open('%s/joints.csv' % args.datadir, 'w')
    fp_train = open('%s/train_joints.csv' % args.datadir, 'w')
    fp_test = open('%s/test_joints.csv' % args.datadir, 'w')
    for img_fn in sorted(glob.glob('%s/images/*.jpg' % args.datadir)):
        index = int(re.search(ur'im([0-9]+)', b(img_fn)).groups()[0]) - 1
        str_j = [str(j) for j in joints[index].flatten().tolist()]

        out_list = [b(img_fn)]
        out_list.extend(str_j)
        out_str = ','.join(out_list)
        print(out_str, file=fp)
        if index in perm:
            print(out_str, file=fp_test)
        else:
            print(out_str, file=fp_train)
    fp.close()
    fp_train.close()
    fp_test.close()
