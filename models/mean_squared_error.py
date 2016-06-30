#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import function
from chainer.cuda import get_array_module
from chainer.utils import type_check

import numpy


class MeanSquaredError(function.Function):

    """Mean squared error (a.k.a. Euclidean loss) function."""

    ignore_label = -1

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, inputs):
        x, t = inputs
        xp = get_array_module(x)
        mask = t != self.ignore_label
        self.count = int(mask.sum())
        self.diff = x * mask - t * mask
        diff = self.diff.ravel()
        return xp.asarray(diff.dot(diff) / self.count, dtype=diff.dtype),

    def backward(self, inputs, gy):
        coeff = gy[0] * gy[0].dtype.type(2. / self.count)
        gx0 = coeff * self.diff
        return gx0, -gx0


def mean_squared_error(x0, x1):
    """Mean squared error function.

    This function computes mean squared error between two variables. The mean
    is taken over the minibatch. Note that the error is not scaled by 1/2.

    """
    return MeanSquaredError()(x0, x1)
