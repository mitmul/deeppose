#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from scipy.io import loadmat

import numpy as np

crop_sizes = {
    '12-oc': (0, 0),
    'along': (0, 0),
    'batma': (0, 0),
    'bend-': (0, 0),
    'ten-c': (0, 0),
    'giant': (42, 396),
    'princ': (10, 464),
    'schin': (6, 461),
    'others': (56, 364)
}


def get_joint_list(joints):
    head = np.asarray(joints['reye']) + \
        np.asarray(joints['leye']) + \
        np.asarray(joints['nose'])
    head /= 3
    del joints['reye']
    del joints['leye']
    del joints['nose']
    joints['head'] = head.tolist()
    joint_pos = [joints['lwri']]
    joint_pos.append(joints['lelb'])
    joint_pos.append(joints['lsho'])
    joint_pos.append(joints['head'])
    joint_pos.append(joints['rsho'])
    joint_pos.append(joints['relb'])
    joint_pos.append(joints['rwri'])

    return np.array(joint_pos).flatten()


def save_crop_images_and_joints():
    training_indices = loadmat('data/FLIC-full/tr_plus_indices.mat')
    training_indices = training_indices['tr_plus_indices'].flatten()

    examples = loadmat('data/FLIC-full/examples.mat')
    examples = examples['examples'][0]
    joint_ids = ['lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri', 'lhip',
                 'lkne', 'lank', 'rhip', 'rkne', 'rank', 'leye', 'reye',
                 'lear', 'rear', 'nose', 'msho', 'mhip', 'mear', 'mtorso',
                 'mluarm', 'mruarm', 'mllarm', 'mrlarm', 'mluleg', 'mruleg',
                 'mllleg', 'mrlleg']

    available = joint_ids[:8]
    available.extend(joint_ids[12:14])
    available.extend([joint_ids[16]])

    target_joints = ['lsho', 'lelb', 'lwri',
                     'leye', 'reye', 'nose',
                     'rsho', 'relb', 'rwri']

    fp_train = open('data/FLIC-full/train_joints.csv', 'w')
    fp_test = open('data/FLIC-full/test_joints.csv', 'w')
    for i, example in enumerate(examples):
        joint = example[2].T
        joint = dict(zip(joint_ids, joint))
        fname = example[3][0]
        joint = get_joint_list(joint)
        msg = '{},{}'.format(fname, ','.join([str(j) for j in joint.tolist()]))
        if i in training_indices:
            print(msg, file=fp_train)
        else:
            print(msg, file=fp_test)


if __name__ == '__main__':
    save_crop_images_and_joints()
