#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import re
import os
import argparse
from chainer import cuda
import cPickle as pickle
import numpy as np
import cv2 as cv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model_file_dir', type=str, default='.')
    args = parser.parse_args()

    if args.gpu >= 0:
        cuda.init()

    for model_file in glob.glob('%s/*.chainermodel' % args.model_file_dir):
        epoch = int(re.search(ur'epoch_([0-9]+)', model_file).groups()[0])
        model = pickle.load(open(model_file, 'rb'))
        model.to_cpu()

        conv1 = model.parameters[0]
        conv1 = conv1.transpose((0, 2, 3, 1))

        n, h, w, c = conv1.shape
        side = int(np.ceil(np.sqrt(n)))
        pad = 2

        canvas1 = np.zeros((h * side + pad * (side + 1),
                            w * side + pad * (side + 1), c))
        for i, feature_map in enumerate(conv1):
            x = w * (i % side) + pad * (i % side + 1)
            y = h * (i / side) + pad * (i / side + 1)
            feature_map -= feature_map.min()
            feature_map /= feature_map.max()
            canvas1[y:y + h, x:x + w] = feature_map

        result_dir = os.path.dirname(args.model_file_dir)
        if result_dir == '':
            result_dir = '.'
        result_dir = '%s/filters' % result_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        cv.imwrite('%s/conv1_epoch_%05d.jpg' %
                   (result_dir, epoch), canvas1 * 255)

    if args.gpu >= 0:
        cuda.shutdown()
