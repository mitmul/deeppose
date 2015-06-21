#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from scipy.misc import imrotate
from scipy.ndimage.interpolation import shift
from sklearn.preprocessing import scale


class Transform(object):

    def __init__(self, **params):
        [setattr(self, key, value) for key, value in params.iteritems()]

    def transform(self, datum, datadir, train=True):
        img_fn = '%s/images/%s' % (datadir, datum[0])
        self._img = cv.imread(img_fn)
        self.orig = self._img.copy()
        self._joints = np.asarray([int(float(p)) for p in datum[1:]])

        if hasattr(self, 'padding'):
            self.crop()
        if hasattr(self, 'flip'):
            self.fliplr()
        if hasattr(self, 'size'):
            self.resize()
        if hasattr(self, 'lcn'):
            self.contrast()

        # joint pos centerization
        h, w, c = self._img.shape
        center_pt = np.array([w / 2, h / 2])  # x,y order
        joints = zip(self._joints[0::2], self._joints[1::2])
        joints = np.array(joints) - center_pt
        joints[:, 0] /= w
        joints[:, 1] /= h
        self._joints = joints.flatten()

        return self._img, self._joints

    def crop(self):
        # image cropping
        joints = self._joints.reshape((len(self._joints) / 2, 2))
        x, y, w, h = cv.boundingRect(np.asarray([joints.tolist()]))

        # bounding rect extending
        inf, sup = self.padding
        r = sup - inf
        pad_w_r = np.random.rand() * r + inf  # inf~sup
        pad_h_r = np.random.rand() * r + inf  # inf~sup
        x -= (w * pad_w_r - w) / 2
        y -= (h * pad_h_r - h) / 2
        w *= pad_w_r
        h *= pad_h_r

        # shifting
        if hasattr(self, 'shift'):
            x += np.random.rand() * self.shift * 2 - self.shift
            y += np.random.rand() * self.shift * 2 - self.shift

        # clipping
        x, y, w, h = [int(z) for z in [x, y, w, h]]
        x = np.clip(x, 0, self._img.shape[1] - 1)
        y = np.clip(y, 0, self._img.shape[0] - 1)
        w = np.clip(w, 1, self._img.shape[1] - (x + 1))
        h = np.clip(h, 1, self._img.shape[0] - (y + 1))
        self._img = self._img[y:y + h, x:x + w]

        # joint shifting
        joints = np.asarray([(j[0] - x, j[1] - y) for j in joints])
        self._joints = joints.flatten()

    def resize(self):
        if not isinstance(self.size, int):
            raise Exception('self.size should be int')
        orig_h, orig_w, _ = self._img.shape
        self._joints[0::2] = self._joints[0::2] / float(orig_w) * self.size
        self._joints[1::2] = self._joints[1::2] / float(orig_h) * self.size
        self._img = cv.resize(self._img, (self.size, self.size),
                              interpolation=cv.INTER_NEAREST)

    def contrast(self):
        if self.lcn:
            if not self._img.dtype == np.float32:
                self._img = self._img.astype(np.float32)
            # local contrast normalization
            for ch in range(self._img.shape[2]):
                im = self._img[:, :, ch]
                im = (im - np.mean(im)) / \
                    (np.std(im) + np.finfo(np.float32).eps)
                self._img[:, :, ch] = im

    def fliplr(self):
        if np.random.randint(2) == 1 and self.flip == True:
            self._img = np.fliplr(self._img)
            self._joints[0:: 2] = self._img.shape[1] - self._joints[0:: 2]

    def revert(self, img, pred):
        h, w, c = img.shape
        center_pt = np.array([w / 2, h / 2])
        joints = np.array(zip(pred[0::2], pred[1::2]))  # x,y order
        joints[:, 0] *= w
        joints[:, 1] *= h
        joints += center_pt

        return joints
