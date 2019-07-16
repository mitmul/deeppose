#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 Shunta Saito

import os
import io
import zipfile

import cv2
import numpy as np
from chainercv.chainer_experimental.datasets import sliceable
from scipy.io import loadmat
from deeppose.utils import flic_utils
import threading


class FLICDataset(sliceable.GetterDataset):

    def __init__(self, split='train', dataset_zip_path='data/FLIC.zip'):
        super().__init__()
        self.dataset_zip_path = dataset_zip_path
        self.zf = zipfile.ZipFile(self.dataset_zip_path)
        self.zf_pid = os.getpid()
        self.img_paths = [fn for fn in self.zf.namelist() if fn.endswith('.jpg')]

        examples = loadmat(io.BytesIO(self.zf.read('FLIC/examples.mat')))['examples'][0]
        if split == 'train':
            self.examples = [e for e in examples if e['istrain'][0][0] == 1]
        elif split == 'test':
            self.examples = [e for e in examples if e['istest'][0][0] == 1]
        else:
            raise ValueError('\'split\' argument should be either \'train\' or \'test\'.')
        
        joint_names = flic_utils.flic_joint_names
        available_joints = flic_utils.flic_available_joints
        self.available_joint_ids = [joint_names.index(a) for a in available_joints] 

        self.add_getter('img', self._get_image)
        self.add_getter('point', self._get_point)
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.examples)

    def __getstate__(self):
        d = self.__dict__.copy()
        d['zf'] = None
        d['lock'] = None
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self._lock = threading.Lock()

    def _get_image(self, i):
        """Extract image from the zipfile.

        Returns:
            img (ndarray): The shape is (C, H, W) and the channel follows RGB order (NOT BGR!).
        """
        with self.lock:
            if self.zf is None or self.zf_pid != os.getpid():
                self.zf_pid = os.getpid()
                self.zf = zipfile.ZipFile(self.dataset_zip_path)
            image_data = self.zf.read('FLIC/images/{}'.format(self.examples[i][3][0]))

        image_file = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_file, cv2.IMREAD_COLOR)
        assert len(img.shape) == 3 and img.shape[2] == 3, "The image has wrong shape: {}".format(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, dtype=np.float32)
        img = img.transpose((2, 0, 1))

        return img

    def _get_point(self, i):
        point = self.examples[i][2].T[self.available_joint_ids].astype(np.float32)
        return point[:, ::-1]  # (x, y) -> (y, x)

