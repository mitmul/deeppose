#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('../../scripts')  # to resume from result dir
import logging
import time
import os
import imp
import shutil
import numpy as np
import cPickle as pickle
from chainer import optimizers, cuda
from transform import Transform
from draw_loss import draw_loss_curve
from progressbar import ProgressBar
from multiprocessing import Process, Queue
from cmd_options import get_arguments


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
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
    if args.restart_from is not None:
        model = pickle.load(open(args.restart_from, 'rb'))
    if args.gpu >= 0:
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
    optimizer.setup(model)

    return model, optimizer


def load_data(trans, args, input_q, data_q):
    c = args.channel
    s = args.size
    d = args.joint_num * 2
    xp = cuda.cupy if args.gpu >= 0 and cuda.available else np

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

    # training
    xp = cuda.cupy if args.gpu >= 0 and cuda.available else np
    for i in range(0, N, args.batchsize):
        input_q.put(train_dl[perm[i:i + args.batchsize]])
    for i in range(0, N, args.batchsize):
        input_data, label = data_q.get()
        input_data = xp.asarray(input_data, dtype=np.float32)
        label = xp.asarray(label, dtype=np.float32)

        optimizer.zero_grads()
        loss, pred = model.forward(input_data, label, train=True)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * args.batchsize
        pbar.update(i + args.batchsize if (i + args.batchsize) < N else N)

    return sum_loss


def eval(test_dl, N, model, trans, args, input_q, data_q):
    pbar = ProgressBar(N)
    sum_loss = 0

    # training
    xp = cuda.cupy if args.gpu >= 0 and cuda.available else np
    for i in xrange(0, N, args.batchsize):
        input_q.put(test_dl[i:i + args.batchsize])
    for i in xrange(0, N, args.batchsize):
        input_data, label = data_q.get(True, None)
        input_data = xp.asarray(input_data, dtype=np.float32)
        label = xp.asarray(label, dtype=np.float32)

        loss, pred = model.forward(input_data, label, train=False)
        sum_loss += float(loss.data) * args.batchsize
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
    args = get_arguments()

    # create result dir
    log_fn, result_dir = create_result_dir(args)

    # create model and optimizer
    model, optimizer = get_model_optimizer(result_dir, args)
    train_dl, test_dl = load_dataset(args)
    N, N_test = len(train_dl), len(test_dl)
    logging.info('# of training data:{}'.format(N))
    logging.info('# of test data:{}'.format(N_test))

    # augmentation setting
    trans = Transform(padding=[args.crop_pad_inf, args.crop_pad_sup],
                      flip=bool(args.flip),
                      size=args.size,
                      shift=args.shift,
                      lcn=bool(args.lcn))

    logging.info(time.strftime('%Y-%m-%d_%H-%M-%S'))
    logging.info('start training...')

    # start training data loading thread
    tinput_q, tdata_q = Queue(), Queue()
    tdata_loader = Process(target=load_data,
                           args=(trans, args, tinput_q, tdata_q))
    tdata_loader.start()

    # start validation data loading thread
    vinput_q, vdata_q = Queue(), Queue()
    vdata_loader = Process(target=load_data,
                           args=(trans, args, vinput_q, vdata_q))
    vdata_loader.start()

    # learning loop
    for epoch in range(1, args.epoch + 1):
        # train
        st = time.time()
        sum_loss = train(train_dl, N, model, optimizer, trans, args, tinput_q,
                         tdata_q)
        msg = get_log_msg('training', epoch, sum_loss, N, args, st)
        logging.info(msg)
        print('\n%s' % msg)

        # eval
        st = time.time()
        sum_loss = eval(test_dl, N_test, model, trans, args, vinput_q, vdata_q)
        msg = get_log_msg('test', epoch, sum_loss, N_test, args, st)
        logging.info(msg)
        print('\n%s' % msg)

        if epoch == 1 or epoch % args.snapshot == 0:
            model_fn = '{}/{}_epoch_{}.chainermodel'.format(
                result_dir, args.prefix, epoch + args.epoch_offset)
            pickle.dump(model, open(model_fn, 'wb'), -1)
        draw_loss_curve(log_fn, '{}/log.jpg'.format(result_dir))

    # quit training data loading thread
    tinput_q.put(None)
    tdata_loader.join()

    # quit data loading thread
    vinput_q.put(None)
    vdata_loader.join()

    model_fn = '%s/%s_epoch_%d.chainermodel' % (
        result_dir, args.prefix, epoch + args.epoch_offset)
    pickle.dump(model, open(model_fn, 'wb'), -1)

    input_q.put(None)
    data_loader.join()

    logging.info(time.strftime('%Y-%m-%d_%H-%M-%S'))
