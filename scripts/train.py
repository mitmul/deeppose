#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import logging
import time
import os
import sys
import imp
import shutil
import numpy as np
from chainer import optimizers, cuda, Variable
import chainer.functions as F
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
    # create result dir
    result_dir = 'results/' + os.path.basename(args.model).split('.')[0]
    result_dir += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
    result_dir += str(time.time()).replace('.', '')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    log_fn = '%s/log.txt' % result_dir
    logging.basicConfig(filename=log_fn, level=logging.DEBUG)
    logging.info(args)

    return log_fn, result_dir


def get_model_optimizer(result_dir, args):
    model_fn = os.path.basename(args.model)
    model_name = model_fn.split('.')[0]
    module = imp.load_source(model_fn.split('.')[0], args.model)
    Net = getattr(module, model_name)

    shutil.copy(args.model, '%s/%s' % (result_dir, model_fn))
    shutil.copy(__file__, '%s/%s' % (result_dir, os.path.basename(__file__)))

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
    optimizer = optimizers.MomentumSGD(lr=0.0005, momentum=0.9)
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
            x, t = trans.transform(x.split(','), args.datadir)
            input_data[j] = x.transpose((2, 0, 1))
            label[j] = t

        data_q.put((input_data, label))


def train(train_dl, N, model, optimizer, trans, args, input_q, data_q):
    global index
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
    parser.add_argument('--model', '-m', type=str, default='cifar10_model')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--prefix', '-p', type=str)
    parser.add_argument('--snapshot', '-s', type=int, default=5)
    parser.add_argument('--restart_from', '-r', type=str)
    parser.add_argument('--epoch_offset', '-o', type=int, default=0)
    parser.add_argument('--datadir', '-d', type=str, default='data/FLIC-full')
    parser.add_argument('--channel', '-c', type=int, default=3)
    parser.add_argument('--size', '-z', type=int, default=224)
    parser.add_argument('--crop_pad_inf', '-i', type=float, default=1.5)
    parser.add_argument('--crop_pad_sup', '-u', type=float, default=2.0)
    parser.add_argument('--joint_num', '-j', type=int, default=7)
    args = parser.parse_args()

    # create result dir
    log_fn, result_dir = create_result_dir(args)

    model, optimizer = get_model_optimizer(result_dir, args)
    train_dl, test_dl = load_dataset(args)
    N = len(train_dl)
    N_test = len(test_dl)
    logging.info('# of training data:{}'.format(N))
    logging.info('# of test data:{}'.format(N_test))

    # augmentation setting
    trans = Transform(padding=[1.5, 2.0],
                      flip=True,
                      size=args.size,
                      shift=5,
                      norm=True)

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
        print(msg)

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
        print(msg)

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
