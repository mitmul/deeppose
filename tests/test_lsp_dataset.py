#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from os.path import basename
from os.path import splitext

import csv
import cv2 as cv
import glob
import numpy as np
import os


def draw_limb(img, joints, i, j, color):
    if ((joints[i][0] > 0) and (joints[j][0] > 0) and
            (joints[i][1] > 0) and (joints[j][1] > 0)):
        cv.line(img, joints[i], joints[j], (255, 255, 255),
                thickness=2, lineType=cv.CV_AA)
        cv.line(img, joints[i], joints[j], color,
                thickness=1, lineType=cv.CV_AA)

    return img


def draw_joints(img, joints):
    h, w, c = img.shape

    img = draw_limb(img, joints, 0, 1, (0, 255, 0))
    img = draw_limb(img, joints, 1, 2, (0, 255, 0))
    img = draw_limb(img, joints, 3, 4, (0, 255, 0))
    img = draw_limb(img, joints, 4, 5, (0, 255, 0))
    img = draw_limb(img, joints, 2, 8, (255, 0, 0))
    img = draw_limb(img, joints, 8, 9, (255, 0, 0))
    img = draw_limb(img, joints, 9, 3, (255, 0, 0))
    img = draw_limb(img, joints, 3, 2, (255, 0, 0))
    img = draw_limb(img, joints, 8, 7, (0, 0, 255))
    img = draw_limb(img, joints, 7, 6, (0, 0, 255))
    img = draw_limb(img, joints, 9, 10, (0, 0, 255))
    img = draw_limb(img, joints, 10, 11, (0, 0, 255))
    img = draw_limb(img, joints, 12, 13, (0, 255, 0))

    # all joint points
    for j, joint in enumerate(joints):
        if joint[0] > 0 and joint[1] > 0:
            cv.circle(img, joint, 5, (0, 0, 255), -1)
            cv.circle(img, joint, 3, (0, 255, 0), -1)
            cv.putText(img, '%d' % j, joint, cv.FONT_HERSHEY_SIMPLEX, 0.3,
                       (0, 0, 0), thickness=3, lineType=cv.CV_AA)
            cv.putText(img, '%d' % j, joint, cv.FONT_HERSHEY_SIMPLEX, 0.3,
                       (255, 255, 255), thickness=1, lineType=cv.CV_AA)

    return img


if __name__ == '__main__':
    out_dir = 'data/lspet_dataset/test_images'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    joints_csv = csv.reader(open('data/lspet_dataset/test_joints.csv'))
    for i, line in enumerate(joints_csv):
        img_fn = 'data/lspet_dataset/images/%s' % line[0]
        img = cv.imread(img_fn)

        joints = [int(float(j)) for j in line[1:]]
        joints = zip(joints[0::2], joints[1::2])

        draw = draw_joints(img, joints)
        cv.imwrite('%s/%s' % (out_dir, basename(img_fn)), draw)

        print(img_fn)
