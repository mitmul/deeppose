#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable, FunctionSet, cuda
import chainer.functions as F
import chainer.functions.basic_math as M


class AlexNetBN_PReLU(FunctionSet):

    """
    VGGnet with Batch Normalization and Parameterized ReLU
    - It works fine with Adam
    """

    def __init__(self):
        super(AlexNetBN_PReLU, self).__init__(
            conv1=F.Convolution2D(3, 96, 11, stride=4, pad=1),
            bn1=F.BatchNormalization(96),
            prelu1=F.PReLU(),
            conv2=F.Convolution2D(96, 256, 5, stride=1, pad=2),
            bn2=F.BatchNormalization(256),
            prelu2=F.PReLU(),
            conv3=F.Convolution2D(256, 384, 3, stride=1, pad=1),
            prelu3=F.PReLU(),
            conv4=F.Convolution2D(384, 384, 3, stride=1, pad=1),
            prelu4=F.PReLU(),
            conv5=F.Convolution2D(384, 256, 3, stride=1, pad=1),
            prelu5=F.PReLU(),
            fc6=F.Linear(9216, 4096),
            prelu6=F.PReLU(),
            fc7=F.Linear(4096, 4096),
            prelu7=F.PReLU(),
            fc8=F.Linear(4096, 14)
        )

    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = self.prelu1(self.bn1(self.conv1(x)))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.local_response_normalization(h)

        h = self.prelu2(self.bn2(self.conv2(h)))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.local_response_normalization(h)

        h = self.prelu3(self.conv3(h))
        h = self.prelu4(self.conv4(h))
        h = self.prelu5(self.conv5(h))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = F.dropout(self.prelu6(self.fc6(h)), train=train, ratio=0.6)
        h = F.dropout(self.prelu7(self.fc7(h)), train=train, ratio=0.6)
        h = self.fc8(h)

        loss = F.mean_squared_error(t, h)

        return loss, h
