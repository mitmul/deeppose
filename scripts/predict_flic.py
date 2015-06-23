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

    return model


def load_data(trans, args, x):
    c = args.channel
    s = args.size
    d = args.joint_num * 2

    # data augmentation
    input_data = np.zeros((len(x), c, s, s))
    label = np.zeros((len(x), d))

    for i, line in enumerate(x):
        d, t = trans.transform(line.split(','), args.datadir, False,
                               args.fname_index, args.joint_index)
        input_data[i] = d.transpose((2, 0, 1))
        label[i] = t

    return input_data, label


def create_tiled_image(perm, out_dir, result_dir, epoch, suffix, N=25):
    fnames = np.array(sorted(glob.glob('%s/*%s.jpg' % (out_dir, suffix))))
    tile_fnames = fnames[perm[:N]]

    h, w, c, pad = 220, 220, 3, 2
    side = int(np.ceil(np.sqrt(len(tile_fnames))))
    canvas = np.ones((side * w + pad * (side + 1),
                      side * h + pad * (side + 1), 3))
    canvas *= 0

    for i, fname in enumerate(tile_fnames):
        img = cv.imread(fname)
        x = w * (i % side) + pad * (i % side) + pad
        y = h * (i / side) + pad * (i / side) + pad
        canvas[y:y + h, x:x + w, :] = img

    if args.resize > 0:
        canvas = cv.resize(canvas, (args.resize, args.resize))
    cv.imwrite('%s/test_%d_tiled_%s.jpg' % (result_dir, epoch, suffix), canvas)


def test(args):
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
    if args.gpu >= 0:
        cuda.init(args.gpu)
    model = load_model(args)
    if args.gpu >= 0:
        model.to_gpu()
    else:
        model.to_cpu()

    # create output dir
    epoch = int(re.search(ur'epoch_([0-9]+)', args.param).groups()[0])
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

        _, preds = model.forward(input_data, labels, train=False)

        if args.gpu >= 0:
            preds = cuda.to_cpu(preds.data)
            input_data = cuda.to_cpu(input_data)
            labels = cuda.to_cpu(labels)

        for n, line in enumerate(lines):
            img_fn = line.split(',')[args.fname_index]
            img = input_data[n].transpose((1, 2, 0))
            pred = preds[n]
            img_pred, pred = trans.revert(img, pred, np.int)

            # turn label data into image coordinates
            label = labels[n]
            img_label, label = trans.revert(img, label, np.int)

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
    epoch = int(re.search(ur'epoch_([0-9]+)', args.param).groups()[0])
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

    if args.mode == 'test':
        test(args)
    elif args.mode == 'tile':
        tile(args)
