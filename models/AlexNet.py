#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import chainer
import chainer.functions as F
import chainer.links as L


class AlexNet(chainer.Chain):

    def __init__(self, n_joints):
        super(AlexNet, self).__init__(
            conv1=L.Convolution2D(3, 96, 11, stride=4, pad=1),
            conv2=L.Convolution2D(96, 256, 5, stride=1, pad=2),
            conv3=L.Convolution2D(256, 384, 3, stride=1, pad=1),
            conv4=L.Convolution2D(384, 384, 3, stride=1, pad=1),
            conv5=L.Convolution2D(384, 256, 3, stride=1, pad=1),
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, n_joints * 2)
        )
        self.train = True

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.local_response_normalization(h)

        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.local_response_normalization(h)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), train=self.train, ratio=0.6)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train, ratio=0.6)

        return self.fc8(h)
