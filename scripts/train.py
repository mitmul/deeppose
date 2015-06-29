#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('../../scripts')  # to resume from result dir
import argparse
import logging
import time
import os
import imp
import shutil
import numpy as np
from chainer import optimizers, cuda
from transform import Transform
import cPickle as pickle
from draw_loss import draw_loss_curve
from progressbar import ProgressBar
from multiprocessing import Process, Queue


def load_dataset(args):
    train_fn = '%s/train_joints.csv' % args.datadir
    test_fn = '%s/test_joints.csv' % args.datadir
    train_dl = np.array([l.strip() for l in open(train_fn).readlines()])
    test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

    return train_dl, test_dl


def create_result_dir(args):
    if args.restart_from is None:
        result_dir = 'results/' + os.path.basename(args.model).split('.')[0]
        result_dir += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
        result_dir += str(time.time()).replace('.', '')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        log_fn = '%s/log.txt' % result_dir
        logging.basicConfig(filename=log_fn, level=logging.DEBUG)
        logging.info(args)
    else:
        result_dir = '.'
        log_fn = 'log.txt'
        logging.basicConfig(filename=log_fn, level=logging.DEBUG)
        logging.info(args)

    return log_fn, result_dir


def get_model_optimizer(result_dir, args):
    model_fn = os.path.basename(args.model)
    model_name = model_fn.split('.')[0]
    module = imp.load_source(model_fn.split('.')[0], args.model)
    Net = getattr(module, model_name)

    dst = '%s/%s' % (result_dir, model_fn)
    if not os.path.exists(dst):
        shutil.copy(args.model, dst)

    dst = '%s/%s' % (result_dir, os.path.basename(__file__))
    if not os.path.exists(dst):
        shutil.copy(__file__, dst)

    # prepare model
    model = Net()
    if args.restart_from is not None:
        if args.gpu >= 0:
            cuda.init(args.gpu)
        model = pickle.load(open(args.restart_from, 'rb'))
    if args.gpu >= 0:
        cuda.init(args.gpu)
        model.to_gpu()

    # prepare optimizer
    if args.opt == 'AdaGrad':
        optimizer = optimizers.AdaGrad(lr=0.0005)
    elif args.opt == 'MomentumSGD':
        optimizer = optimizers.MomentumSGD(lr=0.0005, momentum=0.9)
    elif args.opt == 'Adam':
        optimizer = optimizers.Adam()
    else:
        raise Exception('No optimizer is selected')
    optimizer.setup(model.collect_parameters())

    return model, optimizer


def load_data(trans, args, input_q, data_q):
    c = args.channel
    s = args.size
    d = args.joint_num * 2

    while True:
        x_batch = input_q.get()
        if x_batch is None:
            break

        # data augmentation
        input_data = np.zeros((args.batchsize, c, s, s))
        label = np.zeros((args.batchsize, d))
        for j, x in enumerate(x_batch):
            x, t = trans.transform(x.split(','), args.datadir, True,
                                   args.fname_index, args.joint_index)
            input_data[j] = x.transpose((2, 0, 1))
            label[j] = t

        data_q.put([input_data, label])


def train(train_dl, N, model, optimizer, trans, args, input_q, data_q):
    pbar = ProgressBar(N)
    perm = np.random.permutation(N)
    sum_loss = 0

    # putting all data
    for i in range(0, N, args.batchsize):
        x_batch = train_dl[perm[i:i + args.batchsize]]
        input_q.put(x_batch)

    # training
    for i in range(0, N, args.batchsize):
        input_data, label = data_q.get()

        if args.gpu >= 0:
            input_data = cuda.to_gpu(input_data.astype(np.float32))
            label = cuda.to_gpu(label.astype(np.float32))

        optimizer.zero_grads()
        loss, pred = model.forward(input_data, label, train=True)
        loss.backward()
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        pbar.update(i + args.batchsize if (i + args.batchsize) < N else N)

    return sum_loss


def eval(test_dl, N, model, trans, args, input_q, data_q):
    pbar = ProgressBar(N)
    sum_loss = 0

    # putting all data
    for i in xrange(0, N, args.batchsize):
        x_batch = test_dl[i:i + args.batchsize]
        input_q.put(x_batch)

    # training
    for i in xrange(0, N, args.batchsize):
        input_data, label = data_q.get(True, None)

        if args.gpu >= 0:
            input_data = cuda.to_gpu(input_data.astype(np.float32))
            label = cuda.to_gpu(label.astype(np.float32))

        loss, pred = model.forward(input_data, label, train=False)
        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        pbar.update(i + args.batchsize if (i + args.batchsize) < N else N)

    return sum_loss


def get_log_msg(stage, epoch, sum_loss, N, args, st):
    msg = 'epoch:{:02d}\t{} mean loss={}\telapsed time={} sec'.format(
        epoch + args.epoch_offset,
        stage,
        sum_loss / N,
        time.time() - st)

    return msg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/AlexNet_flic.py',
                        help='model definition file in models dir')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--prefix', type=str, default='AlexNet_flic')
    parser.add_argument('--snapshot', type=int, default=10)
    parser.add_argument('--datadir', type=str, default='data/FLIC-full')
    parser.add_argument('--channel', type=int, default=3)
    parser.add_argument('--flip', type=int, default=1,
                        help='flip left and right for data augmentation')
    parser.add_argument('--size', type=int, default=220,
                        help='resizing')
    parser.add_argument('--crop_pad_inf', type=float, default=1.5,
                        help='random number infimum for padding size when cropping')
    parser.add_argument('--crop_pad_sup', type=float, default=2.0,
                        help='random number supremum for padding size when cropping')
    parser.add_argument('--shift', type=int, default=5,
                        help='slide an image when cropping')
    parser.add_argument('--lcn', type=int, default=1,
                        help='local contrast normalization for data augmentation')
    parser.add_argument('--joint_num', type=int, default=7)
    parser.add_argument('--fname_index', type=int, default=0,
                        help='the index of image file name in a csv line')
    parser.add_argument('--joint_index', type=int, default=1,
                        help='the start index of joint values in a csv line')
    parser.add_argument('--restart_from', type=str, default=None,
                        help='*.chainermodel file path to restart from')
    parser.add_argument('--epoch_offset', type=int, default=0,
                        help='set greater than 0 if you restart from a chainermodel pickle')
    parser.add_argument('--opt', type=str, default='AdaGrad',
                        choices=['AdaGrad', 'MomentumSGD', 'Adam'])
    args = parser.parse_args()

    # create result dir
    log_fn, result_dir = create_result_dir(args)

    # create model and optimizer
    model, optimizer = get_model_optimizer(result_dir, args)
    train_dl, test_dl = load_dataset(args)
    N = len(train_dl)
    N_test = len(test_dl)
    logging.info('# of training data:{}'.format(N))
    logging.info('# of test data:{}'.format(N_test))

    # augmentation setting
    _flip = True if args.flip == 1 else False
    _lcn = True if args.lcn == 1 else False
    trans = Transform(padding=[args.crop_pad_inf, args.crop_pad_sup],
                      flip=_flip,
                      size=args.size,
                      shift=args.shift,
                      lcn=_lcn)

    logging.info(time.strftime('%Y-%m-%d_%H-%M-%S'))
    logging.info('start training...')

    # learning loop
    n_epoch = args.epoch
    batchsize = args.batchsize
    for epoch in range(1, n_epoch + 1):
        # start data loading thread
        input_q = Queue()
        data_q = Queue()
        data_loader = Process(target=load_data,
                              args=(trans, args, input_q, data_q))
        data_loader.start()

        # train
        st = time.time()
        sum_loss = train(train_dl, N, model, optimizer,
                         trans, args, input_q, data_q)
        msg = get_log_msg('training', epoch, sum_loss, N, args, st)
        logging.info(msg)
        print('\n%s' % msg)

        # quit data loading thread
        input_q.put(None)
        data_loader.join()

        # start data loading thread
        input_q = Queue()
        data_q = Queue()
        data_loader = Process(target=load_data,
                              args=(trans, args, input_q, data_q))
        data_loader.start()

        # eval
        st = time.time()
        sum_loss = eval(test_dl, N_test, model,
                        trans, args, input_q, data_q)
        msg = get_log_msg('test', epoch, sum_loss, N_test, args, st)
        logging.info(msg)
        print('\n%s' % msg)

        if epoch == 1 or epoch % args.snapshot == 0:
            model_fn = '%s/%s_epoch_%d.chainermodel' % (
                result_dir, args.prefix, epoch + args.epoch_offset)
            pickle.dump(model, open(model_fn, 'wb'), -1)
        draw_loss_curve(log_fn, '%s/log.jpg' % result_dir)

        # quit data loading thread
        input_q.put(None)
        data_loader.join()

    model_fn = '%s/%s_epoch_%d.chainermodel' % (
        result_dir, args.prefix, epoch + args.epoch_offset)
    pickle.dump(model, open(model_fn, 'wb'), -1)

    input_q.put(None)
    data_loader.join()

    logging.info(time.strftime('%Y-%m-%d_%H-%M-%S'))
