#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import lmdb
import glob
import numpy as np
import cv2 as cv
from scipy.io import loadmat

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


def bounding_rect(joint):
    pt = []
    for n, j in joint.iteritems():
        if np.all(~np.isnan(j)):
            pt.append([int(j[0]), int(j[1])])
    pt = np.asarray([pt])

    return cv.boundingRect(pt)


def crop_image(joint, img):
    x, y, w, h = bounding_rect(joint)
    scale = 1.5
    mscale = 1.5
    x = x - (w * scale - w) / 2 if x - (w * scale - w) / 2 > 0 else 0
    y = y - (h * scale - h) / 2 if y - (h * scale - h) / 2 > 0 else 0
    w = w * scale if x - (w * scale - w) / 2 > 0 else w * mscale
    h = h * scale if y - (h * scale - h) / 2 > 0 else h * mscale

    for n, j in joint.iteritems():
        if np.all(~np.isnan(j)):
            joint[n] = [j[0] - x, j[1] - y]

    return joint, img[y:y + h, x:x + w]


def get_joint_list(joints):
    head = np.asarray(joints['reye']) + \
        np.asarray(joints['leye']) + \
        np.asarray(joints['nose'])
    head /= 3
    del joints['reye']
    del joints['leye']
    del joints['nose']
    joints['head'] = head.tolist()
    joint_pos = joints['lwri']
    joint_pos += joints['lelb']
    joint_pos += joints['lsho']
    joint_pos += joints['head']
    joint_pos += joints['rsho']
    joint_pos += joints['relb']
    joint_pos += joints['rwri']

    return joint_pos


def save_crop_images_and_joints():
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

    if not os.path.exists('data/FLIC-full/crop'):
        os.makedirs('data/FLIC-full/crop')
    if not os.path.exists('data/FLIC-full/joint'):
        os.makedirs('data/FLIC-full/joint')
    for i, example in enumerate(examples):
        joint = example[2].T
        joint = dict(zip(joint_ids, joint))
        fname = example[3][0]
        img = cv.imread('data/FLIC-full/images_c/%s' % fname)

        for k, v in joint.iteritems():
            if np.all(~np.isnan(v)):
                if fname[:5] in crop_sizes.keys():
                    joint[k][1] -= crop_sizes[fname[:5]][0]
                else:
                    joint[k][1] -= crop_sizes['others'][0]
        joint, img = crop_image(joint, img)
        for k, v in joint.iteritems():
            if np.all(~np.isnan(v)):
                joint[k][0] /= img.shape[1]
                joint[k][1] /= img.shape[0]

        img = np.asarray(img, dtype=np.float64)
        img = cv.resize(img, (227, 227))

        out_joints = {}
        for k, v in joint.iteritems():
            if np.all(~np.isnan(v)) and k in target_joints:
                if v[0] < img.shape[1] and v[1] < img.shape[0]:
                    out_joints[k] = v

        if len(out_joints) == len(target_joints):
            cv.imwrite('data/FLIC-full/crop/%s' % fname, img)
            joints = get_joint_list(out_joints)
            np.save('data/FLIC-full/joint/%s' %
                    fname.split('.')[0], joints)
            print('{0:10}\t{1}'.format(i, fname))


def exclude_black_zone():
    if not os.path.exists('data/FLIC-full/images_c'):
        os.mkdir('data/FLIC-full/images_c')
    for fname in glob.glob('data/FLIC-full/images/*.jpg'):
        img = cv.imread(fname)
        pref = os.path.basename(fname)[:5]
        y = None
        h = None
        if pref in crop_sizes.keys():
            y, h = crop_sizes[pref]
            if h == 0:
                h = img.shape[0]
        else:
            y, h = crop_sizes['others']
        img = img[y:y + h, :, :]
        cv.imwrite('data/FLIC-full/images_c/%s' % os.path.basename(fname), img)
        print fname


if __name__ == '__main__':
    exclude_black_zone()
    save_crop_images_and_joints()
