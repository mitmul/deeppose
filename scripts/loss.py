#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import reporter

import chainer


class MeanSquaredError(chainer.Function):

    """Mean squared error (a.k.a. Euclidean loss) function.

    In forward method, it calculates mean squared error between two variables
    with ignoring all elements that the value of ignore_joints at the same
    position is 0.

    """

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[2].dtype == numpy.int32,
            in_types[0].shape == in_types[1].shape,
            in_types[1].shape == in_types[2].shape,
        )

    def forward(self, inputs):
        x, t, ignore = inputs
        xp = chainer.cuda.get_array_module(x)
        self.count = int(ignore.sum())
        self.diff = (x * ignore - t * ignore).astype(xp.float32)
        diff = self.diff.ravel()
        return xp.asarray(diff.dot(diff) / self.count, dtype=xp.float32),

    def backward(self, inputs, gy):
        coeff = gy[0] * gy[0].dtype.type(2. / self.count)
        gx0 = coeff * self.diff
        return gx0, -gx0, None


def mean_squared_error(x0, x1, ignore):
    """Mean squared error function.

    This function computes mean squared error between two variables. The mean
    is taken over the minibatch. Note that the error is not scaled by 1/2.

    """
    return MeanSquaredError()(x0, x1, ignore)


class PoseEstimationError(chainer.Chain):

    def __init__(self, predictor):
        super(PoseEstimationError, self).__init__(predictor=predictor)
        self.lossfun = mean_squared_error
        self.y = None
        self.loss = None

    def __call__(self, *args):
        x, t, ignore = args[:3]
        self.y = None
        self.loss = None
        self.pre_rec = None
        self.y = self.predictor(x)
        self.loss = self.lossfun(self.y, t, ignore)
        reporter.report({'loss': self.loss}, self)
        return self.loss
