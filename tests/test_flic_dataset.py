#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
from os.path import basename, splitext
import cv2 as cv
import numpy as np
import glob


def draw_limb(img, joints, i, j, color):
    cv.line(img, joints[i], joints[j], (255, 255, 255),
            thickness=2, lineType=16)
    cv.line(img, joints[i], joints[j], color,
            thickness=1, lineType=16)

    return img


def draw_joints(img, joints, line=True, text_scale=0.5):
    h, w, c = img.shape

    if line:
        # left hand to left elbow
        img = draw_limb(img, joints, 0, 1, (0, 255, 0))
        img = draw_limb(img, joints, 1, 2, (0, 255, 0))
        img = draw_limb(img, joints, 4, 5, (0, 255, 0))
        img = draw_limb(img, joints, 5, 6, (0, 255, 0))
        img = draw_limb(img, joints, 2, 4, (255, 0, 0))
        neck = tuple((np.array(joints[2]) + np.array(joints[4])) / 2)
        joints.append(neck)
        img = draw_limb(img, joints, 3, 7, (255, 0, 0))
        joints.pop()

    # all joint points
    for j, joint in enumerate(joints):
        cv.circle(img, joint, 5, (0, 0, 255), -1)
        cv.circle(img, joint, 3, (0, 255, 0), -1)
        cv.putText(img, '%d' % j, joint, cv.FONT_HERSHEY_SIMPLEX, text_scale,
                   (0, 0, 0), thickness=3, lineType=16)
        cv.putText(img, '%d' % j, joint, cv.FONT_HERSHEY_SIMPLEX, text_scale,
                   (255, 255, 255), thickness=1, lineType=16)

    return img


if __name__ == '__main__':
    out_dir = 'data/FLIC-full/test_images'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    joints_csv = csv.reader(open('data/FLIC-full/test_joints.csv'))
    for line in joints_csv:
        img_fn = 'data/FLIC-full/images/%s' % line[0]
        img = cv.imread(img_fn)

        joints = [int(float(j)) for j in line[1:]]
        joints = zip(joints[0::2], joints[1::2])

        draw = draw_joints(img, joints)
        cv.imwrite('%s/%s' % (out_dir, basename(img_fn)), draw)

        print img_fn
