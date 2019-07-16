import os
import unittest
from deeppose.models import alexnet
import numpy as np
from chainer import computational_graph

class TestAlexNet(unittest.TestCase):

    def setUp(self):
        self.model = alexnet.AlexNet(n_class=22)
        self.model.pick = 'fc7'
        self.model.remove_unused()

        insize = self.model.insize
        self.x = np.random.rand(1, 3, insize, insize).astype(np.float32)

    def test_forward(self):
        y = self.model(self.x)
        assert y.shape == (1, 4096)

    def test_graph(self):
        y = self.model(self.x)
        g = computational_graph.build_computational_graph(y)

        outdir = 'data/test_models'
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, 'alexnet.dot'), 'w') as o:
            o.write(g.dump())