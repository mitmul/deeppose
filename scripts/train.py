#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('../../scripts')  # to resume from result dir
import re
import six
import logging
import time
import os
import ctypes
import imp
import shutil
import numpy as np
from chainer import Variable, optimizers, cuda, serializers
from transform import Transform
from draw_loss import draw_loss_curve
from multiprocessing import Process, Queue, Array
from cmd_options import get_arguments


def load_dataset(args):
    train_fn = '%s/train_joints.csv' % args.datadir
    test_fn = '%s/test_joints.csv' % args.datadir
    train_dl = np.array([l.strip() for l in open(train_fn).readlines()])
    test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

    return train_dl, test_dl


def create_result_dir(args):
    if args.resume_model is None:
        result_dir = 'results/{}_{}'.format(
            os.path.splitext(os.path.basename(args.model))[0],
            time.strftime('%Y-%m-%d_%H-%M-%S'))
        if os.path.exists(result_dir):
            result_dir += '_{}'.format(np.random.randint(100))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    else:
        result_dir = os.path.dirname(args.resume_model)

    log_fn = '%s/log.txt' % result_dir
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=log_fn, level=logging.DEBUG)
    logging.info(args)

    args.log_fn = log_fn
    args.result_dir = result_dir


def get_model(args):
    model_fn = os.path.basename(args.model)
    model = imp.load_source(model_fn.split('.')[0], args.model).model

    if 'result_dir' in args:
        dst = '%s/%s' % (args.result_dir, model_fn)
        if not os.path.exists(dst):
            shutil.copy(args.model, dst)

        dst = '%s/%s' % (args.result_dir, os.path.basename(__file__))
        if not os.path.exists(dst):
            shutil.copy(__file__, dst)

    # load model
    if args.resume_model is not None:
        serializers.load_hdf5(args.resume_model, model)

    # prepare model
    if args.gpu >= 0:
        model.to_gpu()

    return model


def get_model_optimizer(args):
    model = get_model(args)

    if 'opt' in args:
        # prepare optimizer
        if args.opt == 'AdaGrad':
            optimizer = optimizers.AdaGrad(lr=args.lr)
        elif args.opt == 'MomentumSGD':
            optimizer = optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
        elif args.opt == 'Adam':
            optimizer = optimizers.Adam()
        else:
            raise Exception('No optimizer is selected')

        optimizer.setup(model)

        if args.resume_opt is not None:
            serializers.load_hdf5(args.resume_opt, optimizer)
            args.epoch_offset = int(
                re.search('epoch-([0-9]+)', args.resume_opt).groups()[0])

        return model, optimizer

    else:
        print('No optimizer generated.')
        return model


def transform(args, x_queue, datadir, fname_index, joint_index, o_queue):
    trans = Transform(args)
    while True:
        x = x_queue.get()
        if x is None:
            break
        x, t = trans.transform(x.split(','), datadir, fname_index, joint_index)
        o_queue.put((x.transpose((2, 0, 1)), t))


def load_data(args, input_q, minibatch_q):
    c = args.channel
    s = args.size
    d = args.joint_num * 2

    input_data_base = Array(ctypes.c_float, args.batchsize * c * s * s)
    input_data = np.ctypeslib.as_array(input_data_base.get_obj())
    input_data = input_data.reshape((args.batchsize, c, s, s))

    label_base = Array(ctypes.c_float, args.batchsize * d)
    label = np.ctypeslib.as_array(label_base.get_obj())
    label = label.reshape((args.batchsize, d))

    x_queue, o_queue = Queue(), Queue()
    workers = [Process(target=transform,
                       args=(args, x_queue, args.datadir, args.fname_index,
                             args.joint_index, o_queue))
               for _ in range(args.batchsize)]
    for w in workers:
        w.start()

    while True:
        x_batch = input_q.get()
        if x_batch is None:
            break

        # data augmentation
        for x in x_batch:
            x_queue.put(x)
        j = 0
        while j != len(x_batch):
            a, b = o_queue.get()
            input_data[j] = a
            label[j] = b
            j += 1
        minibatch_q.put([input_data, label])

    for _ in range(args.batchsize):
        x_queue.put(None)
    for w in workers:
        w.join()


def one_epoch(args, model, optimizer, epoch, data, train):
    model.train = True
    sum_loss = 0
    num = 0
    N = len(data)

    input_q, minibatch_q = Queue(), Queue(maxsize=1)
    data_loader = Process(target=load_data,
                          args=(args, input_q, minibatch_q))
    data_loader.start()

    perm = np.random.permutation(N)
    for i in six.moves.range(0, N, args.batchsize):
        input_q.put(data[perm[i:i + args.batchsize]])

    # training
    xp = cuda.cupy if args.gpu >= 0 and cuda.available else np
    for i in six.moves.range(0, N, args.batchsize):
        input_data, label = minibatch_q.get()
        input_data = xp.asarray(input_data, dtype=np.float32)
        label = xp.asarray(label, dtype=np.float32)
        x = Variable(input_data, volatile=not train)
        t = Variable(label, volatile=not train)

        if train:
            optimizer.update(model, x, t)
        else:
            model(x, t)

        sum_loss += float(model.loss.data) * input_data.shape[0]
        num += input_data.shape[0]

        logging.info('loss:{}'.format(sum_loss / num))

    # quit training data loading thread
    input_q.put(None)
    data_loader.join()

    return sum_loss


if __name__ == '__main__':
    args = get_arguments()
    if cuda.available and args.gpu >= 0:
        cuda.get_device(args.gpu).use()

    np.random.seed(args.seed)

    # create result dir
    create_result_dir(args)

    # create model and optimizer
    model, optimizer = get_model_optimizer(args)
    train_dl, test_dl = load_dataset(args)
    N, N_test = len(train_dl), len(test_dl)
    logging.info('# of training data:{}'.format(N))
    logging.info('# of test data:{}'.format(N_test))

    logging.info(time.strftime('%Y-%m-%d_%H-%M-%S'))
    logging.info('start training...')

    # learning loop
    for epoch in range(args.epoch_offset + 1, args.epoch + 1):
        # train
        sum_loss = one_epoch(args, model, optimizer, epoch, train_dl, True)
        logging.info('epoch:{}\ttraining loss:{}'.format(epoch, sum_loss / N))

        if epoch == 1 or epoch % args.test_freq == 0:
            sum_loss = one_epoch(args, model, optimizer, epoch, test_dl, False)
            logging.info('epoch:{}\ttest loss:{}'.format(
                epoch, sum_loss / N_test))

        if epoch == 1 or epoch % args.snapshot == 0:
            model_fn = '{}/epoch-{}.model'.format(args.result_dir, epoch)
            opt_fn = '{}/epoch-{}.state'.format(args.result_dir, epoch)
            serializers.save_hdf5(model_fn, model)
            serializers.save_hdf5(opt_fn, optimizer)

        draw_loss_curve(args.log_fn, '{}/log.png'.format(args.result_dir))
