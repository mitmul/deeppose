#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
from chainer import cuda
import imp
import argparse
from transform import Transform
import cPickle as pickle
from create_panorama import draw_structure
import cv2 as cv

cuda.init()


def load_model(args):
    model_fn = os.path.basename(args.definition_file)
    model_name = model_fn.split('.')[0]
    module = imp.load_source(model_fn.split('.')[0], args.definition_file)

    model = pickle.load(open(args.model_file, 'rb'))
    model.to_cpu()

    return model


def load_data(trans, args, x):
    c = args.channel
    s = args.size
    d = args.joint_num * 2

    # data augmentation
    input_data = np.zeros((1, c, s, s))
    label = np.zeros((1, d))
    x, t = trans.transform(x.split(','), args.data_dir)
    input_data[0] = x.transpose((2, 0, 1))
    label[0] = t

    return input_data, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameter', '-p', type=str)
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--data_dir', '-d', type=str, default='data/FLIC-full')
    parser.add_argument('--crop_pad_inf', '-i', type=float, default=1.5)
    parser.add_argument('--crop_pad_sup', '-u', type=float, default=2.0)
    parser.add_argument('--channel', '-c', type=int, default=3)
    parser.add_argument('--size', '-s', type=int, default=220)
    parser.add_argument('--joint_num', '-j', type=int, default=7)
    args = parser.parse_args()
    print(args)

    # augmentation setting
    trans = Transform(padding=[args.crop_pad_inf, args.crop_pad_sup],
                      flip=False,
                      size=args.size,
                      shift=0,
                      lcn=False)

    # test data
    test_fn = '%s/test_joints.csv' % args.data_dir
    test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

    # load model
    model = load_model(args)
    result_dir = os.path.dirname(args.model_file) + '/test'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for line in test_dl:
        img_fn = line.split(',')[1]
        input_data, label = load_data(trans, args, line)
        loss, pred = model.forward(input_data, label)
        pred = trans.revert(cuda.to_cpu(pred.data))[0]
        label = trans.revert(label)[0]

        img = input_data[0].transpose((1, 2, 0))
        img = np.array(img.copy())
        pred = pred.astype(np.int32).tolist()
        pred = zip(pred[0::2], pred[1::2])
        img = draw_structure(img, pred)
        cv.imwrite('%s/%s' % (result_dir, img_fn), img)
