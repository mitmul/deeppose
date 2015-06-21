#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable, FunctionSet
import chainer.functions as F
import chainer.functions.basic_math as M


class VGG(FunctionSet):

    """
    VGGnet with Batch Normalization and Parameterized ReLU
    - It works fine with Adam
    """

    def __init__(self):
        super(VGG, self).__init__(
            conv1=F.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv2=F.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv3=F.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv4=F.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv5=F.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv6=F.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv7=F.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv8=F.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv9=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv10=F.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv11=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv12=F.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv13=F.Convolution2D(512, 512, 3, stride=1, pad=1),

            fc14=F.Linear(25088, 4096),
            fc15=F.Linear(4096, 4096),
            pred=F.Linear(4096, 28)
        )

    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv11(h))
        h = F.relu(self.conv12(h))
        h = F.relu(self.conv13(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.dropout(F.relu(self.fc14(h)), train=train, ratio=0.5)
        h = F.dropout(F.relu(self.fc15(h)), train=train, ratio=0.5)
        h = self.pred(h)

        loss = F.mean_squared_error(h, t)

        return loss, h
