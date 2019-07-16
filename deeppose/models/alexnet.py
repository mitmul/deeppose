#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 Shunta Saito

import chainer
import chainer.functions as F
import chainer.links as L
from chainercv.links import Conv2DActiv
from chainercv.links import PickableSequentialChain


class AlexNet(PickableSequentialChain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 220

    def __init__(self, n_class=22):
        super().__init__()
        with self.init_scope():
            self.conv1 = Conv2DActiv(None, 96, 11, 4)
            self.lrn1 = _local_responce_normalization
            self.pool1 = _max_pooling_2d

            self.conv2 = Conv2DActiv(None, 256,  5, 2)
            self.lrn2 = _local_responce_normalization
            self.pool2 = _max_pooling_2d

            self.conv3 = Conv2DActiv(None, 384,  3, pad=1)
            self.conv4 = Conv2DActiv(None, 384,  3, pad=1)
            self.conv5 = Conv2DActiv(None, 256,  3, pad=1)
            self.pool5 = _max_pooling_2d

            self.fc6 = L.Linear(None, 4096)
            self.dropout1 = _dropout
            self.fc7 = L.Linear(None, 4096)
            self.dropout2 = _dropout
            self.fc8 = L.Linear(None, n_class)


def _max_pooling_2d(x):
    return F.max_pooling_2d(x, ksize=3, stride=2)


def _local_responce_normalization(x):
    return F.local_response_normalization(x)


def _dropout(x):
    return F.dropout(x, ratio=0.6)