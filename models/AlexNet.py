#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable, FunctionSet
import chainer.functions as F
import chainer.functions.basic_math as M


class AlexNet(FunctionSet):

    """
    VGGnet with Batch Normalization and Parameterized ReLU
    - It works fine with Adam
    """

    def __init__(self):
        super(AlexNet, self).__init__(
            conv1=F.Convolution2D(3, 96, 11, stride=4, pad=1),
            bn1=F.BatchNormalization(96),
            conv2=F.Convolution2D(96, 256, 5, stride=1, pad=2),
            bn2=F.BatchNormalization(256),
            conv3=F.Convolution2D(256, 384, 3, stride=1, pad=1),
            conv4=F.Convolution2D(384, 384, 3, stride=1, pad=1),
            conv5=F.Convolution2D(384, 256, 3, stride=1, pad=1),
            fc6=F.Linear(9216, 4096),
            fc7=F.Linear(4096, 4096),
            fc8=F.Linear(4096, 28)
        )

    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.local_response_normalization(h)

        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.local_response_normalization(h)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), train=train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=train, ratio=0.5)
        h = self.fc8(h)
        h = F.tanh(h)

        loss = F.mean_squared_error(h, t)

        print loss.data

        return loss, h
