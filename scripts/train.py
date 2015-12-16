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
import imp
import shutil
import numpy as np
from chainer import Variable, optimizers, cuda, serializers
from transform import Transform
from draw_loss import draw_loss_curve
from multiprocessing import Process, Queue
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
            optimizer = optimizers.AdaGrad(lr=0.0005)
        elif args.opt == 'MomentumSGD':
            optimizer = optimizers.MomentumSGD(lr=0.0005, momentum=0.9)
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


def one_epoch(args, model, optimizer, epoch, input_q, data_q, train):
    model.train = True
    perm = np.random.permutation(N)
    sum_loss = 0
    num = 0

    # training
    xp = cuda.cupy if args.gpu >= 0 and cuda.available else np
    for i in six.moves.range(0, N, args.batchsize):
        input_q.put(train_dl[perm[i:i + args.batchsize]])
    for i in six.moves.range(0, N, args.batchsize):
        input_data, label = data_q.get()
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

    return sum_loss


if __name__ == '__main__':
    args = get_arguments()

    # create result dir
    create_result_dir(args)

    # create model and optimizer
    model, optimizer = get_model_optimizer(args)
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
    for epoch in range(args.epoch_offset + 1, args.epoch + 1):
        # train
        sum_loss = one_epoch(args, model, optimizer, epoch,
                             tinput_q, tdata_q, True)
        logging.info('epoch:{}\ttraining loss:{}'.format(epoch, sum_loss / N))
        sum_loss = one_epoch(args, model, optimizer, epoch,
                             vinput_q, vdata_q, False)
        logging.info('epoch:{}\ttest loss:{}'.format(epoch, sum_loss / N_test))

        if epoch == 1 or epoch % args.snapshot == 0:
            model_fn = '{}/epoch-{}.model'.format(args.result_dir, epoch)
            opt_fn = '{}/epoch-{}.state'.format(args.result_dir, epoch)
            serializers.save_hdf5(model_fn, model)
            serializers.save_hdf5(opt_fn, optimizer)

        draw_loss_curve(args.log_fn, '{}/log.jpg'.format(args.result_dir))

    # quit training data loading thread
    tinput_q.put(None)
    tdata_loader.join()

    # quit data loading thread
    vinput_q.put(None)
    vdata_loader.join()
