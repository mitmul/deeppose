#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import lmdb
import glob
import numpy as np
import cv2 as cv
import caffe


def get_joint_list(joint_fn):
    joints = np.load(joint_fn)
    joints = dict(joints.tolist())
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


def get_img_datum(image_fn):
    img = cv.imread(image_fn, cv.IMREAD_COLOR)
    img = img.swapaxes(0, 2).swapaxes(1, 2)
    datum = caffe.io.array_to_datum(img, 0)

    return datum


def get_jnt_datum(image_fn):
    base = os.path.basename(image_fn).split('.')[0]
    joint = get_joint_list('data/FLIC-full/joint/%s.npy' % base)
    datum = caffe.io.caffe_pb2.Datum()
    datum.channels = len(joint)
    datum.height = 1
    datum.width = 1
    datum.float_data.extend(joint)

    return datum


def del_and_create(dname):
    if os.path.exists(dname):
        shutil.rmtree(dname)
    os.makedirs(dname)


def create_dataset():
    img_db_fn = 'data/FLIC-full/image_train.lmdb'
    del_and_create(img_db_fn)
    img_env = lmdb.Environment(img_db_fn, map_size=1099511627776)
    img_txn = img_env.begin(write=True, buffers=True)

    jnt_db_fn = 'data/FLIC-full/joint_train.lmdb'
    del_and_create(jnt_db_fn)
    jnt_env = lmdb.Environment(jnt_db_fn, map_size=1099511627776)
    jnt_txn = jnt_env.begin(write=True, buffers=True)

    keys = np.arange(100000)
    np.random.shuffle(keys)

    for i, image_fn in enumerate(glob.glob('data/FLIC-full/crop/*.jpg')):
        img_datum = get_img_datum(image_fn)
        jnt_datum = get_jnt_datum(image_fn)
        key = '%010d' % keys[i]

        img_txn.put(key, img_datum.SerializeToString())
        jnt_txn.put(key, jnt_datum.SerializeToString())

        if i % 10000 == 0:
            img_txn.commit()
            jnt_txn.commit()
            jnt_txn = jnt_env.begin(write=True, buffers=True)
            img_txn = img_env.begin(write=True, buffers=True)

        print i, 'images'

    img_txn.commit()
    jnt_txn.commit()
    img_env.close()
    jnt_env.close()

if __name__ == '__main__':
    create_dataset()
