#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lmdb
import os


def del_and_create(dname):
    if os.path.exists(dname):
        shutil.rmtree(dname)
    os.makedirs(dname)


def get_img_datum(image_fn):
    img = cv.imread(image_fn, cv.IMREAD_COLOR)
    img = img.swapaxes(0, 2).swapaxes(1, 2)
    datum = caffe.io.array_to_datum(img, 0)

    return datum


def get_jnt_datum(image_fn, joint_fn):
    joint = get_joint_list(joint_fn)
    datum = caffe.io.caffe_pb2.Datum()
    datum.channels = len(joint)
    datum.height = 1
    datum.width = 1
    datum.float_data.extend(joint)

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
    img_fns += glob.glob('data/lspet_dataset/images/*.jpg')
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

        print i, 'images'

    img_txn.commit()
    jnt_txn.commit()
    img_env.close()
    jnt_env.close()

if __name__ == '__main__':
    create_dataset()
