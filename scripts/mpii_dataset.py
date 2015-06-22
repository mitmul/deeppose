#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import sys
import json
import random
import cv2 as cv
import numpy as np
from scipy.io import loadmat
import argparse

random.seed(1701)
np.random.seed(1701)

data_dir = 'data/mpii'
img_dir = 'data/mpii/images'
json_fn = 'data/mpii/joints.json'

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

using_joints = [2, 3, 8, 9, 10, 11, 12, 13, 14, 15]


def draw_joints(img, points, head_rect=None):
    points = dict([(k, (int(points[k][0]), int(points[k][1])))
                   for k in points.keys()])

    # lower body
    if 0 in points and 1 in points:
        cv.line(img, points[0], points[1], (255, 100, 100), 3)
    if 1 in points and 2 in points:
        cv.line(img, points[1], points[2], (255, 100, 100), 3)
    if 3 in points and 4 in points:
        cv.line(img, points[3], points[4], (255, 100, 100), 3)
    if 4 in points and 5 in points:
        cv.line(img, points[4], points[5], (255, 100, 100), 3)

    # torso
    if 2 in points and 6 in points:
        cv.line(img, points[2], points[6], (0, 0, 255), 3)
    if 6 in points and 3 in points:
        cv.line(img, points[6], points[3], (0, 0, 255), 3)
    if 2 in points and 12 in points:
        cv.line(img, points[2], points[12], (0, 0, 255), 3)
    if 12 in points and 7 in points:
        cv.line(img, points[12], points[7], (0, 0, 255), 3)
    if 7 in points and 13 in points:
        cv.line(img, points[7], points[13], (0, 0, 255), 3)
    if 13 in points and 3 in points:
        cv.line(img, points[13], points[3], (0, 0, 255), 3)

    # arms
    if 12 in points and 11 in points:
        cv.line(img, points[12], points[11], (0, 255, 0), 3)
    if 11 in points and 10 in points:
        cv.line(img, points[11], points[10], (0, 255, 0), 3)
    if 13 in points and 14 in points:
        cv.line(img, points[13], points[14], (0, 255, 0), 3)
    if 14 in points and 15 in points:
        cv.line(img, points[14], points[15], (0, 255, 0), 3)

    # headrect
    if head_rect:
        p1 = (int(head_rect[0]), int(head_rect[1]))
        p2 = (int(head_rect[2]), int(head_rect[3]))
        cv.rectangle(img, p1, p2, (150, 100, 255), 3)

    # joints
    for k, p in points.iteritems():
        cv.circle(img, p, 5, (0, 255, 0), -1)
        cv.circle(img, p, 3, (0, 0, 255), -1)
        cv.putText(img, '%d' % k, p,
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def del_and_create(dname):
    if os.path.exists(dname):
        shutil.rmtree(dname)
    os.makedirs(dname)


def fix_wrong_joints(joint):
    if '12' in joint and '13' in joint and '2' in joint and '3' in joint:
        if ((joint['12'][0] < joint['13'][0]) and
                (joint['3'][0] < joint['2'][0])):
            joint['2'], joint['3'] = joint['3'], joint['2']
        if ((joint['12'][0] > joint['13'][0]) and
                (joint['3'][0] > joint['2'][0])):
            joint['2'], joint['3'] = joint['3'], joint['2']

    return joint


def save_sample_images():
    out_dir = 'mpii/samples'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i, line in enumerate(open('mpii/joints.json')):
        joints = json.loads(line.strip())
        img = cv.imread('mpii/images/%s' % joints['filename'])
        joint_pos = joints['joint_pos']
        joint_pos = dict([(int(k), v) for k, v in joint_pos.iteritems()])
        head_rect = joints['head_rect']
        draw_joints(img, joint_pos, head_rect)
        cv.imwrite('%s/%s' % (out_dir, joints['filename']), img)
        print i


def save_joints():
    datadir = 'data/mpii'
    train_joints_fn = 'data/mpii/train_joints.csv'
    test_joints_fn = 'data/mpii/test_joints.csv'
    mat = loadmat('%s/mpii_human_pose_v1_u12_1.mat' % datadir)

    fp_train = open(train_joints_fn, 'w')
    fp_test = open(test_joints_fn, 'w')

    train_num = 0
    test_num = 0
    for i, (anno, train_flag) in enumerate(
        zip(mat['RELEASE']['annolist'][0, 0][0],
            mat['RELEASE']['img_train'][0, 0][0])):

        img_fn = anno['image']['name'][0, 0][0]
        train_flag = int(train_flag)

        if 'annopoints' in str(anno['annorect'].dtype):
            annopoints = anno['annorect']['annopoints'][0]
            for annopoint in annopoints:
                if annopoint != []:
                    # joint coordinates
                    annopoint = annopoint['point'][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                    x = [x[0, 0] for x in annopoint['x'][0]]
                    y = [y[0, 0] for y in annopoint['y'][0]]
                    joint_pos = {}
                    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                        joint_pos[str(_j_id)] = [float(_x), float(_y)]
                    joint_pos = fix_wrong_joints(joint_pos)

                    print img_fn, joint_pos
                    print joint_pos.values()
                    print zip(x, y)
                    sys.exit()

    print train_num, test_num
if __name__ == '__main__':
    save_joints()
    # save_sample_images()
