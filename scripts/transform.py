#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2 as cv
import json
import numpy as np
import os


class Transform(object):

    def __init__(self, args):
        self.args = args
        self.swap_joints = json.loads(args.symmetric_joints)

    def transform(self, datum, datadir, fname_index=0, joint_index=1):
        img_fn = '%s/images/%s' % (datadir, datum[fname_index])
        if not os.path.exists(img_fn):
            raise Exception('%s is not exist' % img_fn)

        self.img = cv.imread(img_fn)
        self.joints = np.asarray([int(float(p)) for p in datum[joint_index:]])

        if self.args.cropping == 1:
            self.cropping()
        if self.args.flip == 1:
            self.fliplr()
        if self.args.size > 0:
            self.resize()
        if self.args.gcn == 1:
            self.contrast()

        # joint pos centerization
        h, w, c = self.img.shape
        center_pt = np.array([w / 2, h / 2], dtype=np.float32)  # x,y order
        joints = list(zip(self.joints[0::2], self.joints[1::2]))

        posi_joints = [(j[0], j[1]) for j in joints if j[0] >= 0 and j[1] >= 0]
        x, y, ww, hh = cv.boundingRect(np.asarray([posi_joints]))
        self.bbox = [(x, y), (x + ww, y + hh)]

        joints = np.array(joints, dtype=np.float32) - center_pt
        joints[:, 0] /= w
        joints[:, 1] /= h
        self.joints = joints.flatten()

        return self.img, self.joints

    def cropping(self):
        # image cropping
        joints = self.joints.reshape((len(self.joints) // 2, 2))
        posi_joints = [(j[0], j[1]) for j in joints if j[0] > 0 and j[1] > 0]
        x, y, w, h = cv.boundingRect(np.asarray([posi_joints]))
        if w < self.args.min_dim:
            w = self.args.min_dim
        if h < self.args.min_dim:
            h = self.args.min_dim

        # bounding rect extending
        inf, sup = self.args.crop_pad_inf, self.args.crop_pad_sup
        r = sup - inf
        pad_w_r = np.random.rand() * r + inf  # inf~sup
        pad_h_r = np.random.rand() * r + inf  # inf~sup
        x -= (w * pad_w_r - w) / 2
        y -= (h * pad_h_r - h) / 2
        w *= pad_w_r
        h *= pad_h_r

        # shifting
        x += np.random.rand() * self.args.shift * 2 - self.args.shift
        y += np.random.rand() * self.args.shift * 2 - self.args.shift

        # clipping
        x, y, w, h = [int(z) for z in [x, y, w, h]]
        x = np.clip(x, 0, self.img.shape[1] - 1)
        y = np.clip(y, 0, self.img.shape[0] - 1)
        w = np.clip(w, 1, self.img.shape[1] - (x + 1))
        h = np.clip(h, 1, self.img.shape[0] - (y + 1))
        self.img = self.img[y:y + h, x:x + w]

        # joint shifting
        joints = np.asarray([(j[0] - x, j[1] - y) for j in joints])
        self.joints = joints.flatten()

    def resize(self):
        if not isinstance(self.args.size, int):
            raise TypeError(
                'self.args.size should be int, but {}'.format(
                    type(self.args.size)))
        orig_h, orig_w = self.img.shape[:2]
        self.joints[0::2] = self.joints[0::2] / float(orig_w) * self.args.size
        self.joints[1::2] = self.joints[1::2] / float(orig_h) * self.args.size
        self.img = cv.resize(self.img, (self.args.size, self.args.size),
                             interpolation=cv.INTER_NEAREST)

    def contrast(self):
        if self.args.gcn:
            if not self.img.dtype == np.float32:
                self.img = self.img.astype(np.float32)
            # global contrast normalization
            self.img -= self.img.reshape(-1, 3).mean(axis=0)
            self.img -= self.img.reshape(-1, 3).std(axis=0) + 1e-5
        else:
            self.img /= 255.0

    def fliplr(self):
        if np.random.randint(2) == 1 and self.args.flip:
            self.img = np.fliplr(self.img)
            self.joints[0::2] = self.img.shape[1] - self.joints[0:: 2]
            joints = list(zip(self.joints[0::2], self.joints[1::2]))
            for i, j in self.swap_joints:
                joints[i], joints[j] = joints[j], joints[i]
            self.joints = np.array(joints).flatten()

    def revert(self, img, pred):
        h, w, c = img.shape
        center_pt = np.array([w / 2, h / 2])
        joints = np.array(list(zip(pred[0::2], pred[1::2])))  # x,y order
        joints[:, 0] *= w
        joints[:, 1] *= h
        joints += center_pt
        joints = joints.astype(np.int32)

        if self.args.gcn:
            img -= img.min()
            img /= img.max()
            img *= 255
            img = img.astype(np.uint8)

        return img, joints
