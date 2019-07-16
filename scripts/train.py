#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random

import chainer
import chainer.functions as F
import chainerx
import numpy as np
from chainer import dataset
from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from chainer import training
from chainer.training import extensions
from chainercv import transforms
from deeppose.datasets import flic_dataset
from deeppose.models import alexnet
from deeppose.models import deeppose
from PIL import Image
from deeppose.utils import common
import random

class DeepPoseTrainChain(chainer.Chain):

    def __init__(self, n_point=22):
        super().__init__()
        extractor = alexnet.AlexNet()
        extractor.pick = 'dropout2'  # Extract output from this layer
        extractor.remove_unused()  # Remove subsequent layers
        with self.init_scope():
            self.model = deeppose.DeepPose(extractor=extractor, n_point=n_point)

    def encode(self, point, height, width):
        """Encode joint coordinates into normalized ones for loss calculation."""
        xp = chainer.backend.get_array_module(point)
        b, n_point = point.shape[:2]
        if point.ndim == 2:
            point = F.reshape(point, (b, n_point // 2, 2))
        center = xp.asarray([height / 2, width / 2], dtype=point.dtype)
        
        return (point - center) / xp.asarray([height, width], dtype=point.dtype)

    def decode(self, point, img_shape):
        pass

    def forward(self, x, y):
        _, _, height, width = x.shape
        assert height == self.model.extractor.insize
        assert width == self.model.extractor.insize

        pred = self.model(x)
        norm_pred = self.encode(pred, height, width)
        norm_y = self.encode(y, height, width)
        loss = F.mean_squared_error(norm_pred, norm_y)

        chainer.reporter.report({'loss': loss}, self)

        return loss


class TrainTransform(object):

    def __init__(self, insize=220):
        self.insize = insize
        self.scale_h = 1.5
        self.scale_w = 1.2
        self.random_offset_ratio_y = 0.2
        self.random_offset_ratio_x = 0.2

    def __call__(self, x):
        img, point = x

        img, point = common.crop_with_joints(
            img, point, self.scale_h, self.scale_w, self.random_offset_ratio_y, self.random_offset_ratio_x)
        img, point = common.to_square(img, point, (self.insize, self.insize))

        if random.randint(0, 1) == 1:
            img, point = common.lr_flip(img, point)

        return img, point.astype(np.float32)


class ValidTransform(object):

    def __init__(self, insize=220):
        self.insize = insize
        self.scale_h = 1.5
        self.scale_w = 1.2

    def __call__(self, x):
        img, point = x

        img, point = common.crop_with_joints(img, point, self.scale_h, self.scale_w)
        img, point = common.to_square(img, point, (self.insize, self.insize))

        return img, point.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Training of DeepPose on the FLIC dataset')
    parser.add_argument('--batchsize', '-B', type=int, default=128, help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--resume', '-r', default='', help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='results', help='Output directory')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250, help='Validation minibatch size')
    parser.add_argument('--device', '-d', type=str, default='cuda:0', help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset-zip-path', '-D', type=str, default='data/FLIC.zip')
    parser.add_argument('--loaderjob', '-j', type=int, default=8)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    device = chainer.get_device(args.device)

    # Initialize the model to train
    model = DeepPoseTrainChain(n_point=22)
    model.to_device(device)
    device.use()

    # Load the dataset files
    dataset = flic_dataset.FLICDataset(split='train', dataset_zip_path=args.dataset_zip_path)

    # Split dataset into train and valid
    np.random.seed(0)
    idx = int(np.random.randint(len(dataset) * 0.8))
    train, valid = dataset.slice[:idx], dataset.slice[idx:]
    
    # Apply data augmentation
    train = TransformDataset(train, ('img', 'point'), TrainTransform(model.model.extractor.insize))
    valid = TransformDataset(valid, ('img', 'point'), ValidTransform(model.model.extractor.insize))

    # These iterators load the images with subprocesses running in parallel to the training/validation.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        valid, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
    optimizer = chainer.optimizers.AdaGrad(lr=0.0005)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = ((1, 'iteration') if args.test else (1, 'epoch'))
    log_interval = ((1, 'iteration') if args.test else (1, 'epoch'))

    trainer.extend(extensions.Evaluator(val_iter, model, device=device), trigger=val_interval)
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(model, 'model_epoch_{.updater.epoch}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'lr']), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
