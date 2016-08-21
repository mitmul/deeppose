#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

import chainer
import numpy
import sys
import unittest


class TestMeanSquaredError(unittest.TestCase):

    def setUp(self):
        sys.path.insert(0, 'models')
        global mean_squared_error
        from mean_squared_error import mean_squared_error
        self.x = numpy.random.uniform(-1, 1, (4, 14)).astype(numpy.float32)
        self.t = numpy.random.uniform(-1, 1, (4, 14)).astype(numpy.float32)
        self.t[0, 5] = -1
        self.t[1, 7] = -1
        self.t[2, 9] = -1
        self.t[3, 11] = -1

    def check_forward(self, x0_data, x1_data):
        x0 = chainer.Variable(x0_data)
        x1 = chainer.Variable(x1_data)
        loss = mean_squared_error(x0, x1)
        loss_value = cuda.to_cpu(loss.data)
        self.assertEqual(loss_value.dtype, numpy.float32)
        self.assertEqual(loss_value.shape, ())

        # Compute expected value
        loss_expect = 0.
        size = 0
        for i in numpy.ndindex(self.x.shape):
            if self.t[i] != -1:
                loss_expect += (self.x[i] - self.t[i]) ** 2
                size += 1
        loss_expect /= size

        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    def check_backward(self, x0_data, x1_data):
        gradient_check.check_backward(
            functions.MeanSquaredError(),
            (x0_data, x1_data), None, eps=1e-2)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.t)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))


testing.run_module(__name__, __file__)
