import chainer
import chainer.links as L
import numpy as np
from deeppose import models


class DeepPose(chainer.Chain):

    def __init__(self, extractor, n_point=22):
        super().__init__()
        with self.init_scope():
            self.extractor = extractor
            self.fc = L.Linear(None, n_point)

    def forward(self, x):
        feat = self.extractor(x)
        return self.fc(feat)
