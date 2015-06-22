#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('tests')
import os
import numpy as np
from chainer import cuda
import imp
import argparse
from transform import Transform
import cPickle as pickle
import cv2 as cv
import test_flic_dataset as flic


def load_model(args):
    model_fn = os.path.basename(args.model)
    model_name = model_fn.split('.')[0]
    module = imp.load_source(model_fn.split('.')[0], args.model)

    model = pickle.load(open(args.param, 'rb'))
    model.to_cpu()

    return model


def load_data(trans, args, x):
    c = args.channel
    s = args.size
    d = args.joint_num * 2

    # data augmentation
    input_data = np.zeros((1, c, s, s))
    label = np.zeros((1, d))
    x, t = trans.transform(x.split(','), args.datadir, False,
                           args.fname_index, args.joint_index)
    input_data[0] = x.transpose((2, 0, 1))
    label[0] = t

    return input_data, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='model definition file in models dir')
    parser.add_argument('--param', type=str,
                        help='trained parameters file in result dir')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--datadir', type=str, default='data/FLIC-full')
    parser.add_argument('--channel', type=int, default=3)
    parser.add_argument('--flip', type=bool, default=True,
                        help='flip left and right for data augmentation')
    parser.add_argument('--size', type=int, default=220,
                        help='resizing')
    parser.add_argument('--crop_pad_inf', type=float, default=1.5,
                        help='random number infimum for padding size when cropping')
    parser.add_argument('--crop_pad_sup', type=float, default=2.0,
                        help='random number supremum for padding size when cropping')
    parser.add_argument('--shift', type=int, default=5,
                        help='slide an image when cropping')
    parser.add_argument('--lcn', type=bool, default=True,
                        help='local contrast normalization for data augmentation')
    parser.add_argument('--joint_num', type=int, default=7)
    parser.add_argument('--fname_index', type=int, default=0,
                        help='the index of image file name in a csv line')
    parser.add_argument('--joint_index', type=int, default=1,
                        help='the start index of joint values in a csv line')
    parser.add_argument('--draw_limb', type=bool, default=True,
                        help='whether draw limb line to visualize')
    parser.add_argument('--text_scale', type=float, default=1.0,
                        help='text scale when drawing indices of joints')
    args = parser.parse_args()
    print(args)

    if args.gpu >= 0:
        cuda.init()

    # augmentation setting
    trans = Transform(padding=[args.crop_pad_inf, args.crop_pad_sup],
                      flip=args.flip,
                      size=args.size,
                      shift=args.shift,
                      lcn=args.lcn)

    # test data
    test_fn = '%s/test_joints.csv' % args.datadir
    test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

    # load model
    model = load_model(args)
    result_dir = os.path.dirname(args.param) + '/test'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for line in test_dl:
        img_fn = line.split(',')[args.fname_index]
        input_data, label = load_data(trans, args, line)
        loss, pred = model.forward(input_data, label)

        img = input_data[0].transpose((1, 2, 0))
        pred = cuda.to_cpu(pred.data)[0]
        label = label[0]

        img_pred, pred = trans.revert(img, pred)
        img_pred = np.array(img_pred.copy())
        pred = [tuple(p) for p in pred]
        # img = flic.draw_joints(img_pred, pred, args.draw_limb, args.text_scale)

        joints = pred
        img = img_pred
        text_scale = args.text_scale
        # all joint points
        for j, joint in enumerate(joints):
            cv.circle(img, joint, 5, (0, 0, 255), -1)
            cv.circle(img, joint, 3, (0, 255, 0), -1)
            cv.putText(img, '%d' % j, joint, cv.FONT_HERSHEY_SIMPLEX, text_scale,
                       (0, 0, 0), thickness=3, lineType=cv.CV_AA)
            cv.putText(img, '%d' % j, joint, cv.FONT_HERSHEY_SIMPLEX, text_scale,
                       (255, 255, 255), thickness=1, lineType=cv.CV_AA)

        print(img_fn)
        # img_label, label = trans.revert(img, label)
        # img_label = draw_structure(img_label, pred)

        cv.imwrite('%s/%s' % (result_dir, img_fn), img)
        # cv.imwrite('%s/%s' % (result_dir, img_fn), img)
