#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 Shunta Saito

import os
import sys
import tempfile
import unittest

import cv2
import numpy as np
from chainercv.utils import write_image
from deeppose.datasets import flic_dataset
from deeppose.utils import flic_utils
from deeppose.utils import common

class TestFLICDataset(unittest.TestCase):

    def setUp(self):
        self.ds = flic_dataset.FLICDataset()
        self.outdir = 'data/flic_test'
        os.makedirs(self.outdir, exist_ok=True)

    def test_get_example(self):
        np.random.seed(0)
        for i in range(10):
            j = np.random.randint(len(self.ds))
            img, point = self.ds[j]

            assert img.shape == (3, 480, 720)
            assert point.shape == (11, 2)

            img = flic_utils.draw_joints(img, point)[:, :, ::-1]
            cv2.imwrite('{}/flic_test_{:02d}.png'.format(self.outdir, i), img)

            img, point = self.ds[j]
            img, point = common.crop_with_joints(img, point)
            vis = flic_utils.draw_joints(img, point)[:, :, ::-1]
            cv2.imwrite('{}/flic_test_crop_{:02d}.png'.format(self.outdir, i), vis)

            img, point = self.ds[j]
            img, point = common.crop_with_joints(img, point, random_offset_ratio_y=0.2, random_offset_ratio_x=0.2)
            vis = flic_utils.draw_joints(img, point)[:, :, ::-1]
            cv2.imwrite('{}/flic_test_offset_{:02d}.png'.format(self.outdir, i), vis)

            img, point = common.to_square(img, point)
            vis = flic_utils.draw_joints(img, point)[:, :, ::-1]
            cv2.imwrite('{}/flic_test_resize_{:02d}.png'.format(self.outdir, i), vis)

            img, point = common.lr_flip(img, point)
            vis = flic_utils.draw_joints(img, point)[:, :, ::-1]
            cv2.imwrite('{}/flic_test_flip_{:02d}.png'.format(self.outdir, i), vis)
