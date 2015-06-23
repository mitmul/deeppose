#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2 as cv
import csv

joint_name = {
    0: 'right_ankle',
    1: 'right_knee',
    2: 'right_hip',
    3: 'left_hip',
    4: 'left_knee',
    5: 'left_ankle',
    6: 'pelvis',
    7: 'clavicle',
    8: 'neck',
    9: 'head_top',
    10: 'right_wrist',
    11: 'right_elbow',
    12: 'right_shoulder',
    13: 'left_shoulder',
    14: 'left_elbow',
    15: 'left_wrist'
}


def draw_joints(img, points, head_rect=None):
    # lower body
    if len(points) > 5:
        cv.line(img, points[0], points[1], (255, 100, 100), 3)
        cv.line(img, points[1], points[2], (255, 100, 100), 3)
        cv.line(img, points[3], points[4], (255, 100, 100), 3)
        cv.line(img, points[4], points[5], (255, 100, 100), 3)

    # torso
    if len(points) > 13:
        cv.line(img, points[2], points[6], (0, 0, 255), 3)
        cv.line(img, points[6], points[3], (0, 0, 255), 3)
        cv.line(img, points[2], points[12], (0, 0, 255), 3)
        cv.line(img, points[12], points[7], (0, 0, 255), 3)
        cv.line(img, points[7], points[13], (0, 0, 255), 3)
        cv.line(img, points[13], points[3], (0, 0, 255), 3)

    # arms
    if len(points) > 15:
        cv.line(img, points[12], points[11], (0, 255, 0), 3)
        cv.line(img, points[11], points[10], (0, 255, 0), 3)
        cv.line(img, points[13], points[14], (0, 255, 0), 3)
        cv.line(img, points[14], points[15], (0, 255, 0), 3)

    # head
    if len(points) > 9:
        cv.line(img, points[8], points[9], (0, 0, 255), 3)

    # joints
    for k, p in enumerate(points):
        cv.circle(img, p, 5, (0, 255, 0), -1)
        cv.circle(img, p, 3, (0, 0, 255), -1)
        cv.putText(img, '%d' % k, p,
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return img

if __name__ == '__main__':
    out_dir = 'data/mpii/test_images'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for line in open('data/mpii/test_joints.csv'):
        datum = line.strip().split(',')
        img_fn = datum[0]
        img = cv.imread('data/mpii/images/%s' % img_fn)
        joints = [int(float(j)) for j in datum[1:]]
        joints = zip(joints[0::2], joints[1::2])

        img = draw_joints(img, joints)
        cv.imwrite('%s/%s' % (out_dir, img_fn), img)
        print img_fn
