#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import re
import sys
import caffe
import numpy as np
import cv2 as cv


def save_tiles(W, fname):
    channels = W.shape[1]
    height = W.shape[2]
    width = W.shape[3]
    print 'channels:', channels
    print 'height:', height
    print 'width:', width

    pad = 1
    side = int(np.ceil(np.sqrt(W.shape[0])))
    output = np.zeros((side * height + (side + 1) * pad,
                       side * width + (side + 1) * pad, channels))
    for sy in range(side):
        for sx in range(side):
            i = sy * side + sx
            if i < W.shape[0]:
                image = W[i].swapaxes(0, 2).swapaxes(0, 1)
                image -= image.min()
                image /= image.max()
                output[sy * height + pad * (sy + 1):
                       (sy + 1) * height + pad * (sy + 1),
                       sx * width + pad * (sx + 1):
                       (sx + 1) * width + pad * (sx + 1), :] = image

    cv.imwrite(fname, output * 255)


def search_dirs():
    for fn in glob.glob('*'):
        if os.path.isdir(fn):
            os.chdir(fn)
            fn = '.'
            define = '%s/train_test.prototxt' % fn
            for model in glob.glob('%s/*.caffemodel' % fn):
                num = re.search(ur'_([0-9]+)\.', model).groups()[0]
                if not os.path.exists('%s/weight_%s.png' % (fn, num)):
                    print define, model
                    net = caffe.Net(define, model)
                    conv1_W = net.params['conv1'][0].data
                    save_tiles(conv1_W, '%s/weight_conv1_%s.png' % (fn, num))
            os.chdir('../')

if __name__ == '__main__':
    define = 'train_test.prototxt'
    for model in glob.glob('snapshots/*.caffemodel'):
        num = re.search(ur'_([0-9]+)\.', model).groups()[0]
        if not os.path.exists('weights/weight_%s.png' % num):
            print define, model
            net = caffe.Net(define, model)
            conv1_W = net.params['conv1'][0].data
            save_tiles(conv1_W, 'weights/weight_%s.png' % num)
