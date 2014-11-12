#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lmdb
import shutil
import glob
import cv2 as cv
import os
import numpy as np
import caffe
import struct


def del_and_create(dname):
    if os.path.exists(dname):
        shutil.rmtree(dname)
    os.makedirs(dname)


def get_img_datum(image_fn):
    img = cv.imread(image_fn, cv.IMREAD_COLOR)
    img = img.swapaxes(0, 2).swapaxes(1, 2)
    datum = caffe.io.array_to_datum(img, 0)

    return datum


def get_jnt_datum(joint_fn):
    joint = np.load(joint_fn)
    datum = caffe.io.caffe_pb2.Datum()
    datum.channels = len(joint)
    datum.height = 1
    datum.width = 1
    datum.float_data.extend(joint.tolist())

    return datum


def create_dataset():
    img_db_fn = 'data/image_train.lmdb'
    del_and_create(img_db_fn)
    img_env = lmdb.Environment(img_db_fn, map_size=1099511627776)
    img_txn = img_env.begin(write=True, buffers=True)

    jnt_db_fn = 'data/joint_train.lmdb'
    del_and_create(jnt_db_fn)
    jnt_env = lmdb.Environment(jnt_db_fn, map_size=1099511627776)
    jnt_txn = jnt_env.begin(write=True, buffers=True)

    keys = np.arange(100000)
    np.random.shuffle(keys)

    img_fns = glob.glob('data/FLIC-full/crop/*.jpg')
    img_fns += glob.glob('data/lspet_dataset/crop/*.jpg')
    jnt_fns = glob.glob('data/FLIC-full/joint/*.npy')
    jnt_fns += glob.glob('data/lspet_dataset/joint/*.npy')
    for i, (img_fn, jnt_fn) in enumerate(
            zip(sorted(img_fns), sorted(jnt_fns))):
        img_datum = get_img_datum(img_fn)
        jnt_datum = get_jnt_datum(jnt_fn)
        key = '%010d' % keys[i]

        img_txn.put(key, img_datum.SerializeToString())
        jnt_txn.put(key, jnt_datum.SerializeToString())

        if i % 10000 == 0:
            img_txn.commit()
            jnt_txn.commit()
            jnt_txn = jnt_env.begin(write=True, buffers=True)
            img_txn = img_env.begin(write=True, buffers=True)

        print i, os.path.basename(img_fn), os.path.basename(jnt_fn)

    img_txn.commit()
    jnt_txn.commit()
    img_env.close()
    jnt_env.close()


def read_test():
    img_db_fn = 'data/image_train.lmdb'
    img_env = lmdb.Environment(img_db_fn, map_size=1099511627776)
    img_txn = img_env.begin(write=True, buffers=True)
    img_cur = img_txn.cursor()

    jnt_db_fn = 'data/joint_train.lmdb'
    jnt_env = lmdb.Environment(jnt_db_fn, map_size=1099511627776)
    jnt_txn = jnt_env.begin(write=True, buffers=True)
    jnt_cur = jnt_txn.cursor()

    for _ in range(10000):
        img_cur.next()
        jnt_cur.next()

    img_datum = caffe.io.caffe_pb2.Datum()
    jnt_datum = caffe.io.caffe_pb2.Datum()

    if not os.path.exists('data/test'):
        os.makedirs('data/test')
    for i in range(100):
        img_key, img_value = img_cur.item()
        jnt_key, jnt_value = jnt_cur.item()
        if img_key != jnt_key:
            sys.exit('img_key and jnt_key should be same')

        img_datum.ParseFromString(img_value)
        jnt_datum.ParseFromString(jnt_value)

        img_data = [struct.unpack('B', d) for d in img_datum.data]
        img_data = np.asarray(img_data, dtype=np.uint8)
        img_data = img_data.reshape(
            (img_datum.channels, img_datum.height, img_datum.width))
        img = np.array(img_data.swapaxes(0, 2).swapaxes(0, 1))

        cv.imwrite('data/test/%d.jpg' % i, img)
        img = cv.imread('data/test/%d.jpg' % i)
        jnt_data = np.asarray(jnt_datum.float_data).reshape((7, 2))
        for j in jnt_data:
            jt = (int(j[0] * img.shape[1]), int(j[1] * img.shape[0]))
            print jt
            cv.circle(img, jt, 5, (0, 0, 255), -1)
        cv.imwrite('data/test/%d.jpg' % i, img)

        img_cur.next()
        jnt_cur.next()

        print i

    img_env.close()
    jnt_env.close()

if __name__ == '__main__':
    # read_test()
    create_dataset()
