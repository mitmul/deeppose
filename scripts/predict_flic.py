#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('tests')
import glob
import re
import os
import numpy as np
from chainer import cuda
import imp
import argparse
from transform import Transform
import cPickle as pickle
import cv2 as cv
from test_flic_dataset import draw_joints


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


def tile(perm, test_dir, result_dir, epoch, suffix, N=25):
    fnames = np.array(glob.glob('%s/*_%s.jpg' % (out_dir, suffix)))
    tile_fnames = fnames[perm[:N]]

    h, w, c, pad = 220, 220, 3, 2
    side = int(np.ceil(np.sqrt(len(tile_fnames))))
    canvas = np.ones((side * w + pad * (side + 1),
                      side * h + pad * (side + 1), 3))
    canvas[:, :, 1] *= 255

    for i, fname in enumerate(tile_fnames):
        img = cv.imread(fname)
        x = w * (i % side) + pad * (i % side) + pad
        y = h * (i / side) + pad * (i / side) + pad
        canvas[y:y + h, x:x + w, :] = img

    cv.imwrite('%s/tiled_%s_%d.jpg' % (result_dir, suffix, epoch), canvas)


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

    # create output dir
    epoch = int(re.search(ur'epoch_([0-9]+)', args.param).groups()[0])
    result_dir = os.path.dirname(args.param)
    out_dir = '%s/test_%d' % (result_dir, epoch)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    mean_error = 0.0
    for i, line in enumerate(test_dl):
        # data loading
        img_fn = line.split(',')[args.fname_index]
        input_data, label = load_data(trans, args, line)

        # prediction
        loss, pred = model.forward(input_data, label)
        img = input_data[0].transpose((1, 2, 0))
        pred = cuda.to_cpu(pred.data)[0]
        img_pred, pred = trans.revert(img, pred, np.int)

        # turn label data into image coordinates
        label = label[0]
        img_label, label = trans.revert(img, label, np.int)

        # calc mean_error
        error = np.linalg.norm(pred - label) / len(pred)
        mean_error += error

        img_pred = np.array(img_pred.copy())
        img_label = np.array(img_label.copy())

        pred = [tuple(p) for p in pred]
        label = [tuple(p) for p in label]

        # all limbs
        draw_joints(img_label, label, args.draw_limb, args.text_scale)
        draw_joints(img_pred, pred, args.draw_limb, args.text_scale)

        print('{}\terror:{:.4f}\tmean_error:{:.4f}'.format(
            img_fn, error, mean_error / (i + 1)))

        fn, ext = os.path.splitext(img_fn)
        cv.imwrite('%s/%s_pred_%.4f%s' %
                   (out_dir, fn, error, ext), img_pred)
        cv.imwrite('%s/%s_label%s' % (out_dir, fn, ext), img_label)

    # save tiled image of randomly chosen results and labels
    perm = np.random.permutation(i - 1)
    tile(perm, out_dir, result_dir, epoch, 'pred', N=25)
    tile(perm, out_dir, result_dir, epoch, 'label', N=25)
