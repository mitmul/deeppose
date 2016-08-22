#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import cuda
from chainer import serializers
from chainer import Variable
from transform import Transform

import argparse
import cv2 as cv
import glob
import imp
import numpy as np
import os
import re
import sys


def cropping(img, joints, min_dim):
    # image cropping
    _joints = joints.reshape((len(joints) // 2, 2))
    posi_joints = [(j[0], j[1]) for j in _joints if j[0] > 0 and j[1] > 0]
    x, y, w, h = cv.boundingRect(np.asarray([posi_joints]))
    if w < min_dim:
        w = min_dim
    if h < min_dim:
        h = min_dim

    # bounding rect extending
    x -= (w * 1.5 - w) / 2
    y -= (h * 1.5 - h) / 2
    w *= 1.5
    h *= 1.5

    # clipping
    x, y, w, h = [int(z) for z in [x, y, w, h]]
    x = np.clip(x, 0, img.shape[1] - 1)
    y = np.clip(y, 0, img.shape[0] - 1)
    w = np.clip(w, 1, img.shape[1] - (x + 1))
    h = np.clip(h, 1, img.shape[0] - (y + 1))
    img = img[y:y + h, x:x + w]

    # joint shifting
    _joints = np.asarray([(j[0] - x, j[1] - y) for j in _joints])
    joints = _joints.flatten()

    return img, joints


def resize(img, joints, size):
    orig_h, orig_w = img.shape[:2]
    joints[0::2] = joints[0::2] / float(orig_w) * size
    joints[1::2] = joints[1::2] / float(orig_h) * size
    img = cv.resize(img, (size, size), interpolation=cv.INTER_NEAREST)

    return img, joints


def contrast(img):
    if not img.dtype == np.float32:
        img = img.astype(np.float32)
    # global contrast normalization
    img -= img.reshape(-1, 3).mean(axis=0)
    img -= img.reshape(-1, 3).std(axis=0) + 1e-5

    return img


def input_transform(datum, datadir, fname_index, joint_index, min_dim, gcn):
    img_fn = '%s/images/%s' % (datadir, datum[fname_index])
    if not os.path.exists(img_fn):
        raise IOError('%s is not exist' % img_fn)

    img = cv.imread(img_fn)
    joints = np.asarray([int(float(p)) for p in datum[joint_index:]])
    img, joints = cropping(img, joints, min_dim)
    img, joints = resize(img, joints, size)
    if gcn:
        img = contrast(img)
    else:
        img /= 255.0

    return img, joints


def load_model(args):
    model_fn = os.path.basename(args.model)
    model_name = model_fn.split('.')[0]
    model = imp.load_source(model_name, args.model)
    model = getattr(model, model_name)
    model = model(args.joint_num)
    serializers.load_npz(args.param, model)
    model.train = False

    return model


def load_data(trans, args, x):
    c = args.channel
    s = args.size
    d = args.joint_num * 2

    # data augmentation
    input_data = np.zeros((len(x), c, s, s))
    label = np.zeros((len(x), d))

    for i, line in enumerate(x):
        d, t = trans.transform(line.split(','), args.datadir,
                               args.fname_index, args.joint_index)
        input_data[i] = d.transpose((2, 0, 1))
        label[i] = t

    return input_data, label


def create_tiled_image(perm, out_dir, result_dir, epoch, suffix, N=25):
    fnames = np.array(sorted(glob.glob('%s/*%s.jpg' % (out_dir, suffix))))
    tile_fnames = fnames[perm[:N]]

    h, w, pad = 220, 220, 2
    side = int(np.ceil(np.sqrt(len(tile_fnames))))
    canvas = np.zeros((side * h + pad * (side + 1),
                       side * w + pad * (side + 1), 3))

    for i, fname in enumerate(tile_fnames):
        img = cv.imread(fname)
        x = w * (i % side) + pad * (i % side + 1)
        y = h * (i // side) + pad * (i // side + 1)
        canvas[y:y + h, x:x + w, :] = img

    if args.resize > 0:
        canvas = cv.resize(canvas, (args.resize, args.resize))
    cv.imwrite('%s/test_%d_tiled_%s.jpg' % (result_dir, epoch, suffix), canvas)


def test(args):
    # test data
    test_fn = '%s/test_joints.csv' % args.datadir
    test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

    # load model
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
    model = load_model(args)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # create output dir
    epoch = int(re.search('epoch-([0-9]+)', args.param).groups()[0])
    result_dir = os.path.dirname(args.param)
    out_dir = '%s/test_%d' % (result_dir, epoch)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_log = '%s.log' % out_dir
    fp = open(out_log, 'w')

    mean_error = 0.0
    N = len(test_dl)
    for i in range(0, N, args.batchsize):
        lines = test_dl[i:i + args.batchsize]
        input_data, labels = load_data(trans, args, lines)

        if args.gpu >= 0:
            input_data = cuda.to_gpu(input_data.astype(np.float32))
            labels = cuda.to_gpu(labels.astype(np.float32))
        else:
            input_data = input_data.astype(np.float32)
            labels = labels.astype(np.float32)

        x = Variable(input_data, volatile=True)
        t = Variable(labels, volatile=True)
        model(x, t)

        if args.gpu >= 0:
            preds = cuda.to_cpu(model.pred.data)
            input_data = cuda.to_cpu(input_data)
            labels = cuda.to_cpu(labels)
        else:
            preds = model.pred.data

        for n, line in enumerate(lines):
            img_fn = line.split(',')[args.fname_index]
            img = input_data[n].transpose((1, 2, 0))
            pred = preds[n]
            img_pred, pred = trans.revert(img, pred)

            # turn label data into image coordinates
            label = labels[n]
            img_label, label = trans.revert(img, label)

            # calc mean_error
            error = np.linalg.norm(pred - label) / len(pred)
            mean_error += error

            # create pred, label tuples
            img_pred = np.array(img_pred.copy())
            img_label = np.array(img_label.copy())
            pred = [tuple(p) for p in pred]
            label = [tuple(p) for p in label]

            # all limbs
            img_label = draw_joints(
                img_label, label, args.draw_limb, args.text_scale)
            img_pred = draw_joints(
                img_pred, pred, args.draw_limb, args.text_scale)

            msg = '{:5}/{:5} {}\terror:{}\tmean_error:{}'.format(
                i + n, N, img_fn, error, mean_error / (i + n + 1))
            print(msg, file=fp)
            print(msg)

            fn, ext = os.path.splitext(img_fn)
            tr_fn = '%s/%d-%d_%s_pred%s' % (out_dir, i, n, fn, ext)
            la_fn = '%s/%d-%d_%s_label%s' % (out_dir, i, n, fn, ext)
            cv.imwrite(tr_fn, img_pred)
            cv.imwrite(la_fn, img_label)


def tile(args):
    # create output dir
    epoch = int(re.search('epoch-([0-9]+)', args.param).groups()[0])
    result_dir = os.path.dirname(args.param)
    out_dir = '%s/test_%d' % (result_dir, epoch)
    if not os.path.exists(out_dir):
        raise Exception('%s is not exist' % out_dir)

    # save tiled image of randomly chosen results and labels
    n_img = len(glob.glob('%s/*pred*' % (out_dir)))
    perm = np.random.permutation(n_img)
    create_tiled_image(perm, out_dir, result_dir, epoch, 'pred', args.n_imgs)
    create_tiled_image(perm, out_dir, result_dir, epoch, 'label', args.n_imgs)


if __name__ == '__main__':
    sys.path.append('tests')
    sys.path.append('models')

    from test_flic_dataset import draw_joints

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='model definition file in models dir')
    parser.add_argument('--param', type=str,
                        help='trained parameters file in result dir')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--datadir', type=str, default='data/FLIC-full')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['test', 'tile'],
                        help='test or create tiled image')
    parser.add_argument('--n_imgs', type=int, default=9,
                        help='how many images will be tiled')
    parser.add_argument('--resize', type=int, default=-1,
                        help='resize the results of tiling')
    parser.add_argument('--seed', type=int, default=9,
                        help='random seed to select images to be tiled')
    parser.add_argument('--draw_limb', type=bool, default=True,
                        help='whether draw limb line to visualize')
    parser.add_argument('--text_scale', type=float, default=1.0,
                        help='text scale when drawing indices of joints')
    args = parser.parse_args()

    result_dir = os.path.dirname(args.param)
    log_fn = grep.grep('{}/log.txt'.format(result_dir))[0]
    for line in open(log_fn):
        if 'Namespace' in line:
            args.joint_num = int(
                re.search('joint_num=([0-9]+)', line).groups()[0])
            args.fname_index = int(
                re.search('fname_index=([0-9]+)', line).groups()[0])
            args.joint_index = int(
                re.search('joint_index=([0-9]+)', line).groups()[0])
            break

    if args.mode == 'test':
        test(args)
    elif args.mode == 'tile':
        np.random.seed(args.seed)
        tile(args)
