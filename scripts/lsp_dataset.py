#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2 as cv
from scipy.io import loadmat


def get_roi(img, joint):
    nonzero_jt = []
    for j in joint:
        if np.all(j > 0):
            nonzero_jt.append(j)
    nonzero_jt = np.asarray(nonzero_jt)

    min_x = int(np.min(nonzero_jt[:, 0]))
    min_y = int(np.min(nonzero_jt[:, 1]))
    max_x = int(np.max(nonzero_jt[:, 0]))
    max_y = int(np.max(nonzero_jt[:, 1]))

    width = (max_x - min_x)
    height = (max_y - min_y)
    ext_width = width * 1.5
    ext_height = height * 1.5

    st_y = min_y + height / 2 - ext_height / 2
    st_y = st_y if st_y > 0 else 0

    en_y = min_y + height / 2 + ext_height / 2
    en_y = en_y if en_y > 0 else img.shape[0]

    st_x = min_x + width / 2 - ext_width / 2
    st_x = st_x if st_x > 0 else 0

    en_x = min_x + width / 2 + ext_width / 2
    en_x = en_x if en_x > 0 else img.shape[1]
    img = img[st_y:en_y, st_x:en_x]

    return img, (st_x, st_y), (en_x - st_x, en_y - st_y)


def get_target_joints(joint, lt, shape):
    joint = joint.tolist()
    joint_pos = []
    joint_pos += joint[11][:2]  # lwri
    joint_pos += joint[10][:2]  # lelb
    joint_pos += joint[9][:2]  # lsho
    head = (np.asarray(joint[12][:2]) +
            np.asarray(joint[13][:2])) / 2
    joint_pos += head.tolist()
    joint_pos += joint[8][:2]  # rsho
    joint_pos += joint[7][:2]  # relb
    joint_pos += joint[6][:2]  # rwri
    joint_pos = np.asarray(joint_pos).reshape((7, 2))
    joint_pos -= np.array([lt[0], lt[1]])
    joint_pos /= np.array([shape[0], shape[1]])

    return joint_pos


def save_crop_images_and_joints():
    jnt_fn = 'data/lspet_dataset/joints.mat'
    joints = loadmat(jnt_fn)
    joints = joints['joints'].swapaxes(0, 2).swapaxes(1, 2)

    if not os.path.exists('data/lspet_dataset/crop'):
        os.mkdir('data/lspet_dataset/crop')
    if not os.path.exists('data/lspet_dataset/mark'):
        os.mkdir('data/lspet_dataset/mark')
    if not os.path.exists('data/lspet_dataset/joint'):
        os.mkdir('data/lspet_dataset/joint')

    img_dir = 'data/lspet_dataset/images'
    for i, joint in enumerate(joints):
        img = cv.imread('%s/im%05d.jpg' % (img_dir, i + 1))
        img, lt, shape = get_roi(img, joint)
        joint_pos = get_target_joints(joint, lt, shape)

        if np.all(joint_pos > 0):
            img = cv.resize(img, (227, 227))
            cv.imwrite('data/lspet_dataset/crop/im%05d.jpg' % (i + 1), img)
            for j in joint_pos:
                p = (int(j[0] * 227), int(j[1] * 227))
                cv.circle(img, p, 5, (0, 0, 255), -1)
            cv.imwrite('data/lspet_dataset/mark/im%05d.jpg' % (i + 1), img)
            joint_pos = joint_pos.flatten()
            np.save('data/lspet_dataset/joint/im%05d' % (i + 1), joint_pos)
            print i


if __name__ == '__main__':
    save_crop_images_and_joints()
