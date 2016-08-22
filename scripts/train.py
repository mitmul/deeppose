#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import iterators
from chainer import optimizers
from chainer import serializers
from chainer import training
from chainer.training import extensions

import chainer
import cmd_options
import dataset
import imp
import logger
import logging
import loss
import os
import shutil
import sys
import tempfile
import time


def create_result_dir(model_path, resume_model):
    if not os.path.exists('results'):
        os.mkdir('results')
    if resume_model is None:
        prefix = '{}_{}'.format(
            os.path.splitext(os.path.basename(model_path))[0],
            time.strftime('%Y-%m-%d_%H-%M-%S'))
        result_dir = tempfile.mkdtemp(prefix=prefix, dir='results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    else:
        result_dir = os.path.dirname(resume_model)

    return result_dir


def create_logger(args, result_dir):
    logging.basicConfig(filename='{}/log.txt'.format(result_dir))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    msg_format = '%(asctime)s [%(levelname)s] %(message)s'
    formatter = logging.Formatter(msg_format)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    logging.info(sys.version_info)
    logging.info('chainer version: {}'.format(chainer.__version__))
    logging.info('cuda: {}, cudnn: {}'.format(
        chainer.cuda.available, chainer.cuda.cudnn_enabled))
    logging.info(args)


def get_model(model_path, n_joints, result_dir, resume_model):
    model_fn = os.path.basename(model_path)
    model_name = model_fn.split('.')[0]
    model = imp.load_source(model_name, model_path)
    model = getattr(model, model_name)

    # Initialize
    model = model(n_joints)

    # Copy files
    dst = '{}/{}'.format(result_dir, model_fn)
    if not os.path.exists(dst):
        shutil.copy(model_path, dst)

    # load model
    if resume_model is not None:
        serializers.load_npz(resume_model, model)

    return model


def get_optimizer(model, opt, lr, adam_alpha=None, adam_beta1=None,
                  adam_beta2=None, adam_eps=None, weight_decay=None,
                  resume_opt=None):
    if opt == 'MomentumSGD':
        optimizer = optimizers.MomentumSGD(lr=lr, momentum=0.9)
    elif opt == 'Adam':
        optimizer = optimizers.Adam(
            alpha=adam_alpha, beta1=adam_beta1,
            beta2=adam_beta2, eps=adam_eps)
    elif opt == 'AdaGrad':
        optimizer = optimizers.AdaGrad(lr=lr)
    elif opt == 'RMSprop':
        optimizer = optimizers.RMSprop(lr=lr)
    else:
        raise Exception('No optimizer is selected')

    # The first model as the master model
    optimizer.setup(model)

    if opt == 'MomentumSGD':
        optimizer.add_hook(
            chainer.optimizer.WeightDecay(weight_decay))

    if resume_opt is not None:
        serializers.load_npz(resume_opt, optimizer)

    return optimizer


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


if __name__ == '__main__':
    args = cmd_options.get_arguments()
    result_dir = create_result_dir(args.model, args.resume_model)
    create_logger(args, result_dir)
    model = get_model(args.model, args.n_joints, result_dir, args.resume_model)
    model = loss.PoseEstimationError(model)
    opt = get_optimizer(model, args.opt, args.lr, adam_alpha=args.adam_alpha,
                        adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2,
                        adam_eps=args.adam_eps, weight_decay=args.weight_decay,
                        resume_opt=args.resume_opt)
    train_dataset = dataset.PoseDataset(
        args.train_csv_fn, args.img_dir, args.im_size, args.fliplr,
        args.rotate, args.rotate_range, args.zoom, args.base_zoom,
        args.zoom_range, args.translate, args.translate_range, args.min_dim,
        args.coord_normalize, args.gcn, args.n_joints, args.fname_index,
        args.joint_index, args.symmetric_joints, args.ignore_label
    )
    test_dataset = dataset.PoseDataset(
        args.test_csv_fn, args.img_dir, args.im_size, args.fliplr,
        args.rotate, args.rotate_range, args.zoom, args.base_zoom,
        args.zoom_range, args.translate, args.translate_range, args.min_dim,
        args.coord_normalize, args.gcn, args.n_joints, args.fname_index,
        args.joint_index, args.symmetric_joints, args.ignore_label
    )

    train_iter = iterators.MultiprocessIterator(train_dataset, args.batchsize)
    test_iter = iterators.MultiprocessIterator(
        test_dataset, args.batchsize, repeat=False, shuffle=False)

    gpus = [int(i) for i in args.gpus.split(',')]
    devices = {'main': gpus[0]}
    if len(gpus) > 2:
        for gid in gpus[1:]:
            devices.update({'gpu{}'.format(gid): gid})
    updater = training.ParallelUpdater(train_iter, opt, devices=devices)

    interval = (args.snapshot, 'epoch')
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=result_dir)
    trainer.extend(extensions.dump_graph('main/loss'))

    # Save parameters and optimization state
    trainer.extend(extensions.snapshot_object(
        model, 'epoch-{.updater.epoch}.model'), trigger=interval)
    trainer.extend(extensions.snapshot_object(
        opt, 'epoch-{.updater.epoch}.state'), trigger=interval)
    trainer.extend(extensions.snapshot(), trigger=interval)

    if args.opt == 'MomentumSGD' or args.opt == 'AdaGrad':
        trainer.reporter.add_observer('lr', opt.lr)
        trainer.extend(IntervalShift(
            'lr', args.lr, args.lr_decay_freq, args.lr_decay_ratio))

    # Show log
    trainer.extend(
        extensions.LogReport(trigger=(args.show_log_iter, 'iteration')))
    trainer.extend(logger.LogPrinter(
        ['epoch', 'main/loss', 'validation/main/loss', 'lr']))

    eval_model = model.copy()
    eval_model.predictor.train = False
    trainer.extend(
        extensions.Evaluator(test_iter, eval_model, device=gpus[0]),
        trigger=(args.valid_freq, 'epoch'))

    trainer.run()
