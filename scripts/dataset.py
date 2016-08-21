#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer.dataset import dataset_mixin
from skimage import transform

import csv
import cv2 as cv
import json
import numpy as np
import os


class PoseDataset(dataset_mixin.DatasetMixin):

    def __init__(self, csv_fn, img_dir, im_size, fliplr, rotate, rotate_range,
                 zoom, base_zoom, zoom_range, translate, translate_range,
                 min_dim, coord_normalize, gcn, joint_num, fname_index,
                 joint_index, symmetric_joints):
        self.csv_fn = csv_fn
        self.img_dir = img_dir
        self.im_size = im_size
        self.fliplr = fliplr
        self.rotate = rotate
        self.rotate_range = rotate_range
        self.zoom = zoom
        self.base_zoom = base_zoom
        self.zoom_range = zoom_range
        self.translate = translate
        self.translate_range = translate_range
        self.min_dim = min_dim
        self.coord_normalize = coord_normalize
        self.gcn = gcn
        self.joint_num = joint_num
        self.fname_index = fname_index
        self.joint_index = joint_index
        self.symmetric_joints = json.loads(symmetric_joints)
        self.load_images()

    def line_to_coords(self, line, joint_index):
        return [float(c) for c in line[joint_index:]]

    def deflatten(self, coords):
        return np.array(list(zip(coords[0::2], coords[1::2])))

    def calc_joint_center(self, joints):
        x_center = (np.min(joints[:, 0]) + np.max(joints[:, 0])) / 2
        y_center = (np.min(joints[:, 1]) + np.max(joints[:, 1])) / 2
        return [x_center, y_center]

    def calc_bbox(self, joints):
        lt = np.min(joints, axis=0)
        rb = np.max(joints, axis=0)
        return [lt[0], lt[1], rb[0], rb[1]]

    def load_images(self):
        self.images = {}
        self.joints = []
        for line in csv.reader(open(self.csv_fn)):
            img_fn = '{}/{}'.format(self.img_dir, line[self.fname_index])
            assert os.path.exists(img_fn), \
                'File not found: {}'.format(img_fn)
            image = cv.imread(img_fn)
            coords = self.line_to_coords(line, self.joint_index)
            joints = self.deflatten(coords)

            # Ignore small label regions smaller than min_dim
            x1, y1, x2, y2 = self.calc_bbox(joints)
            joint_bbox_w, joint_bbox_h = x2 - x1, y2 - y1
            if joint_bbox_w < self.min_dim or joint_bbox_h < self.min_dim:
                continue

            if line[self.fname_index] not in self.images:
                self.images[line[self.fname_index]] = image
            self.joints.append((line[self.fname_index], joints))

    def __len__(self):
        return len(self.images)

    def apply_fliplr(self, image, joints):
        image = cv.flip(image, 1)
        joints[:, 0] = (image.shape[1] - 1) - joints[:, 0]
        for i, j in self.symmetric_joints:
            joints[i], joints[j] = joints[j].copy(), joints[i].copy()
        return image, joints

    def apply_zoom(self, image, joints, fx=None, fy=None):
        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
        joint_vecs = joints - np.array([center_x, center_y])
        if fx is None and fy is None:
            zoom = 1.0 + np.random.uniform(-self.zoom_range, self.zoom_range)
            fx, fy = zoom, zoom
        image = cv.resize(image, None, fx=fx, fy=fy)
        joint_vecs *= np.array([fx, fy])
        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
        joints = joint_vecs + np.array([center_x, center_y])

        return image, joints

    def apply_translate(self, image, joints):
        self.center_x, self.center_y = self.calc_joint_center(joints)
        dx = np.random.randint(-self.translate_range, self.translate_range)
        dy = np.random.randint(-self.translate_range, self.translate_range)
        if dx > 0:
            tmp = np.zeros_like(image)
            tmp[:, dx:] = image[:, :image.shape[1] - dx]
            image = tmp
        else:
            tmp = np.zeros_like(image)
            tmp[:, :image.shape[1] + dx] = image[:, -dx:]
            image = tmp
        if dy > 0:
            tmp = np.zeros_like(image)
            tmp[dy:, :] = image[:image.shape[0] - dy, :]
            image = tmp
        else:
            tmp = np.zeros_like(image)
            tmp[:image.shape[0] + dy, :] = image[-dy:, :]
            image = tmp
        joints += np.array([dx, dy])
        return image, joints

    def apply_rotate(self, image, joints):
        angle = np.random.randint(0, self.rotate_range)
        center = self.calc_joint_center(joints)
        image = transform.rotate(image, angle, center=center)
        image = (image * 255).astype(np.uint8)
        theta = -np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
        center = np.array([image.shape[1] / 2, image.shape[0] / 2])
        joints = rot_mat.dot((joints - center).T).T + center
        return image, np.array(joints.tolist())

    def crop_reshape(self, image, joints):
        if hasattr(self, 'center_x') and hasattr(self, 'center_y'):
            center_x, center_y = self.center_x, self.center_y
        else:
            center_x, center_y = self.calc_joint_center(joints)
        out_size = int(self.im_size * self.base_zoom)
        y_min = np.clip(center_y - out_size // 2, 0, image.shape[0])
        y_max = np.clip(center_y + out_size // 2, 0, image.shape[0])
        x_min = np.clip(center_x - out_size // 2, 0, image.shape[1])
        x_max = np.clip(center_x + out_size // 2, 0, image.shape[1])
        image = image[y_min:y_max, x_min:x_max]
        joints -= np.array([x_min, y_min])
        fx, fy = self.im_size / image.shape[1], self.im_size / image.shape[0]
        image, joints = self.apply_zoom(image, joints, fx, fy)
        return image, joints

    def apply_coord_normalize(self, image, joints):
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        joints -= np.array([center_x, center_y])
        joints[:, 0] /= w
        joints[:, 1] /= h
        return image, joints

    def apply_gcn(self, image, joints):
        image = image.astype(np.float)
        image -= image.reshape(-1, 3).mean(axis=0)
        image /= image.reshape(-1, 3).std(axis=0) + 1e-5
        return image, joints

    def get_example(self, i):
        img_id, joints = self.joints[i]
        image = self.images[img_id]

        ignore_joints = [0 if x == -1 or y == -1 else 1
                         for i, (x, y) in enumerate(joints)]

        if self.fliplr and np.random.randint(0, 2) == 1:
            image, joints = self.apply_fliplr(image, joints)
        if self.zoom:
            image, joints = self.apply_zoom(image, joints)
        if self.translate:
            image, joitns = self.apply_translate(image, joints)
        if self.rotate:
            image, joints = self.apply_rotate(image, joints)

        image, joints = self.crop_reshape(image, joints)

        if self.coord_normalize:
            image, joints = self.apply_coord_normalize(image, joints)
        if self.gcn:
            image, joints = self.apply_gcn(image, joints)

        image = image.astype(np.float32)
        joints = joints.astype(np.float32)
        ignore_joints = np.array(ignore_joints, dtype=np.int32)

        return image, joints, ignore_joints
