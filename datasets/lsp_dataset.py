#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from os.path import basename as b
from scipy.io import loadmat

import argparse
import glob
import numpy as np
import re

if __name__ == '__main__':
    # to fix test set
    np.random.seed(1701)

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data/lspet_dataset')
    args = parser.parse_args()
    print(args)

    jnt_fn = '%s/joints.mat' % args.datadir
    joints = loadmat(jnt_fn)
    joints = joints['joints'].transpose(2, 0, 1)
    joints = joints[:, :, :2]

    N_test = int(len(joints) * 0.1)
    perm = np.random.permutation(int(len(joints)))[:N_test].tolist()

    fp_train = open('%s/train_joints.csv' % args.datadir, 'w')
    fp_test = open('%s/test_joints.csv' % args.datadir, 'w')
    for img_fn in sorted(glob.glob('%s/images/*.jpg' % args.datadir)):
        index = int(re.search('im([0-9]+)', b(img_fn)).groups()[0]) - 1
        str_j = [str(j) if j > 0 else '-1'
                 for j in joints[index].flatten().tolist()]

        out_list = [b(img_fn)]
        out_list.extend(str_j)
        out_str = ','.join(out_list)

        if index in perm:
            print(out_str, file=fp_test)
        else:
            print(out_str, file=fp_train)
    fp_train.close()
    fp_test.close()
