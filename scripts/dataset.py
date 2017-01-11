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
import logging
import numpy as np
import os


class PoseDataset(dataset_mixin.DatasetMixin):

    def __init__(self, csv_fn, img_dir, im_size, fliplr, rotate, rotate_range,
                 zoom, base_zoom, zoom_range, translate, translate_range,
                 min_dim, coord_normalize, gcn, joint_num, fname_index,
                 joint_index, symmetric_joints, ignore_label):
        for key, val in locals().items():
            setattr(self, key, val)
        self.symmetric_joints = json.loads(symmetric_joints)
        self.load_images()
        logging.info('{} is ready'.format(csv_fn))

    def get_available_joints(self, joints, ignore_joints):
        _joints = []
        for i, joint in enumerate(joints):
            if ignore_joints is not None \
                    and (ignore_joints[i][0] == 0 or ignore_joints[i][1] == 0):
                continue
            _joints.append(joint)
        return np.array(_joints)

    def calc_joint_center(self, joints):
        x_center = (np.min(joints[:, 0]) + np.max(joints[:, 0])) / 2
        y_center = (np.min(joints[:, 1]) + np.max(joints[:, 1])) / 2
        return [x_center, y_center]

    def calc_joint_bbox_size(self, joints):
        lt = np.min(joints, axis=0)
        rb = np.max(joints, axis=0)
        return rb[0] - lt[0], rb[1] - lt[1]

    def load_images(self):
        self.images = {}
        self.joints = []
        self.info = []
        for line in csv.reader(open(self.csv_fn)):
            image_id = line[self.fname_index]
            if image_id in self.images:
                image = self.images[image_id]
            else:
                img_fn = '{}/{}'.format(self.img_dir, image_id)
                assert os.path.exists(img_fn), \
                    'File not found: {}'.format(img_fn)
                image = cv.imread(img_fn)
                self.images[image_id] = image

            coords = [float(c) for c in line[self.joint_index:]]
            joints = np.array(list(zip(coords[0::2], coords[1::2])))

            # Ignore small label regions smaller than min_dim
            ig = [0 if v == self.ignore_label else 1 for v in joints.flatten()]
            ig = np.array(list(zip(ig[0::2], ig[1::2])))
            available_joints = self.get_available_joints(joints, ig)
            bbox_w, bbox_h = self.calc_joint_bbox_size(available_joints)
            if bbox_w < self.min_dim or bbox_h < self.min_dim:
                continue

            self.joints.append((image_id, joints))
            center_x, center_y = self.calc_joint_center(available_joints)
            self.info.append((ig, bbox_w, bbox_h, center_x, center_y))

    def __len__(self):
        return len(self.joints)

    def apply_fliplr(self, image, joints):
        image = cv.flip(image, 1)
        joints[:, 0] = (image.shape[1] - 1) - joints[:, 0]
        for i, j in self.symmetric_joints:
            joints[i], joints[j] = joints[j].copy(), joints[i].copy()
        return image, joints

    def apply_zoom(self, image, joints, center_x, center_y, fx=None, fy=None):
        joint_vecs = joints - np.array([center_x, center_y])
        if fx is None and fy is None:
            zoom = 1.0 + np.random.uniform(-self.zoom_range, self.zoom_range)
            fx, fy = zoom, zoom
        image = cv.resize(image, None, fx=fx, fy=fy)
        joint_vecs *= np.array([fx, fy])
        center_x, center_y = center_x * fx, center_y * fy
        joints = joint_vecs + np.array([center_x, center_y])
        return image, joints, center_x, center_y

    def apply_translate(self, image, joints):
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

    def apply_rotate(self, image, joints, ignore_joints):
        available_joints = self.get_available_joints(joints, ignore_joints)
        joint_center = self.calc_joint_center(available_joints)
        angle = np.random.randint(0, self.rotate_range)
        image = transform.rotate(image, angle, center=joint_center)
        image = (image * 255).astype(np.uint8)
        theta = -np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.matrix([[c, -s], [s, c]])
        joints = rot_mat.dot((joints - joint_center).T).T + joint_center
        return image, np.array(joints.tolist())

    def crop_reshape(self, image, joints, bbox_w, bbox_h, center_x, center_y):
        bbox_h, bbox_w = bbox_h * self.base_zoom, bbox_w * self.base_zoom
        y_min = int(np.clip(center_y - bbox_h / 2, 0, image.shape[0]))
        y_max = int(np.clip(center_y + bbox_h / 2, 0, image.shape[0]))
        x_min = int(np.clip(center_x - bbox_w / 2, 0, image.shape[1]))
        x_max = int(np.clip(center_x + bbox_w / 2, 0, image.shape[1]))
        image = image[y_min:y_max, x_min:x_max]
        joints -= np.array([x_min, y_min])
        fx, fy = self.im_size / image.shape[1], self.im_size / image.shape[0]
        cx, cy = image.shape[1] // 2, image.shape[0] // 2
        image, joints = self.apply_zoom(image, joints, cx, cy, fx, fy)[:2]
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
        ignore_joints, bbox_w, bbox_h, cx, cy = self.info[i]

        if self.rotate:
            image, joints = self.apply_rotate(image, joints, ignore_joints)
        if self.translate:
            image, joitns = self.apply_translate(image, joints)
        if self.zoom:
            image, joints, cx, cy = self.apply_zoom(image, joints, cx, cy)

        image, joints = self.crop_reshape(
            image, joints, bbox_w, bbox_h, cx, cy)

        if self.fliplr and np.random.randint(0, 2) == 1:
            image, joints = self.apply_fliplr(image, joints)
        if self.coord_normalize:
            image, joints = self.apply_coord_normalize(image, joints)
        if self.gcn:
            image, joints = self.apply_gcn(image, joints)

        image = image.astype(np.float32).transpose(2, 0, 1)
        joints = joints.astype(np.float32).flatten()
        ignore_joints = np.array(ignore_joints, dtype=np.int32).flatten()

        return image, joints, ignore_joints
