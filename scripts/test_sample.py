#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import caffe
import glob
import cv2 as cv
import numpy as np
import time
import random

img_list = glob.glob('../../data/FLIC-full/crop/*.jpg')
random.shuffle(img_list)

# road caffe net
_, fn = sorted([(int(fn.split('_')[-1].split('.')[0]), fn)
                for fn in glob.glob('snapshots/*.caffemodel')])[-1]
net = caffe.Net('predict.prototxt', fn)
net.set_mode_gpu()

for i, img_fn in enumerate(img_list):
    img = cv.imread(img_fn)
    img = cv.resize(img, (227, 227))
    img = img.swapaxes(0, 2).swapaxes(1, 2)

    net.blobs['data'].data[:, :, :, :] = img
    st = time.clock()
    joints = net.forward().values()[0].flatten()
    print time.clock() - st, 'sec'
    joints = np.asarray(joints)

    # img = cv.imread(img_fn)
    # img = cv.resize(img, (227, 227))
    # for joint in joints.reshape((7, 2)):
    #     joint = (int(joint[0] * 227), int(joint[1] * 227))
    #     cv.circle(img, joint, 5, (0, 0, 255), -1)
    # cv.imwrite('%s_joint.jpg' % img_fn, img)
    if i > 100:
        break
