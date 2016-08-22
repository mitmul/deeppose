#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016 Shunta Saito

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2 as cv
import numpy as np
import os
import sys
import tempfile
import unittest


class TestPoseDataset(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        self.joints = np.array([
            [300.20, 220.20],
            [280.80, 240.40],
            [300.20, 260.60],
            [320.30, 240.40]
        ])
        self.line = '*,'
        self.line += ','.join([str(v) for v in self.joints.flatten().tolist()])
        self.n_test = 2

        # Prepare for FLIC dataset
        fd, self.flic_csv = tempfile.mkstemp()
        with os.fdopen(fd, 'w') as f:
            lines = open('data/FLIC-full/test_joints.csv').readlines()
            np.random.shuffle(lines)
            for line in lines[:self.n_test]:
                print(line.strip(), file=f)
        self.dataset = self.create_dataset(csv_fn=self.flic_csv)

        # Prepare for LSP dataset
        fd, self.lsp_csv = tempfile.mkstemp()
        with os.fdopen(fd, 'w') as f:
            lines = open('data/lspet_dataset/test_joints.csv').readlines()
            np.random.shuffle(lines)
            for line in lines[:self.n_test]:
                print(line.strip(), file=f)

        # Prepare for MPII dataset
        fd, self.mpii_csv = tempfile.mkstemp()
        with os.fdopen(fd, 'w') as f:
            lines = open('data/mpii/test_joints.csv').readlines()
            np.random.shuffle(lines)
            for line in lines[:self.n_test]:
                print(line.strip(), file=f)

    def create_dataset(self,
                       csv_fn='data/FLIC-full/test_joints.csv',
                       img_dir='data/FLIC-full/images',
                       symmetric_joints='[[2, 4], [1, 5], [0, 6]]',
                       im_size=220,
                       fliplr=False,
                       rotate=False,
                       rotate_range=10,
                       zoom=False,
                       base_zoom=1.5,
                       zoom_range=0.2,
                       translate=False,
                       translate_range=5,
                       min_dim=0,
                       coord_normalize=False,
                       gcn=False,
                       joint_num=7,
                       fname_index=0,
                       joint_index=1,
                       ignore_label=-1):
        sys.path.insert(0, 'scripts')
        from dataset import PoseDataset
        dataset = PoseDataset(
            csv_fn, img_dir, im_size, fliplr, rotate, rotate_range, zoom,
            base_zoom, zoom_range, translate, translate_range, min_dim,
            coord_normalize, gcn, joint_num, fname_index, joint_index,
            symmetric_joints, ignore_label
        )
        return dataset

    def test_calc_joint_center(self):
        center = self.dataset.calc_joint_center(self.joints)
        np.testing.assert_array_equal(center, [300.55, 240.4])

    def test_calc_joint_bbox_size(self):
        bbox_w, bbox_h = self.dataset.calc_joint_bbox_size(self.joints)
        self.assertEqual(bbox_w, (320.30 - 280.80))
        self.assertEqual(bbox_h, (260.60 - 220.20))

    def draw_joints(self, image, joints, prefix, ignore_joints):
        if image.shape[2] != 3:
            _image = image.transpose(1, 2, 0).copy()
        else:
            _image = image.copy()
        if joints.ndim == 1:
            joints = np.array(list(zip(joints[0::2], joints[1::2])))
        if ignore_joints.ndim == 1:
            ignore_joints = np.array(
                list(zip(ignore_joints[0::2], ignore_joints[1::2])))
        for i, (x, y) in enumerate(joints):
            if ignore_joints is not None \
                    and (ignore_joints[i][0] == 0 or ignore_joints[i][1] == 0):
                continue
            cv.circle(_image, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv.putText(
                _image, str(i), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 3)
            cv.putText(
                _image, str(i), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 1)
        _, fn_img = tempfile.mkstemp()
        basename = os.path.basename(fn_img)
        fn_img = fn_img.replace(basename, prefix + basename)
        fn_img = fn_img + '.png'
        cv.imwrite(fn_img, _image)

    def test_apply_fliplr(self):
        for i, (img_id, joints) in enumerate(self.dataset.joints):
            image = self.dataset.images[img_id]
            ig, bbox_w, bbox_h, center_x, center_y = self.dataset.info[i]
            self.draw_joints(image, joints, 'fliplr_{}_before_'.format(i), ig)
            image, joints = self.dataset.apply_fliplr(image, joints)
            self.draw_joints(image, joints, 'fliplr_{}_after'.format(i), ig)

    def test_apply_zoom(self):
        for i, (img_id, joints) in enumerate(self.dataset.joints):
            image = self.dataset.images[img_id]
            ig, bbox_w, bbox_h, cx, cy = self.dataset.info[i]
            self.draw_joints(image, joints, 'zoom_{}_before_'.format(i), ig)
            image, joints = self.dataset.apply_zoom(image, joints, cx, cy)[:2]
            self.draw_joints(image, joints, 'zoom_{}_after_'.format(i), ig)

    def test_apply_translate(self):
        for i, (img_id, joints) in enumerate(self.dataset.joints):
            image = self.dataset.images[img_id]
            ig, bbox_w, bbox_h, center_x, center_y = self.dataset.info[i]
            self.draw_joints(image, joints, 'trans_{}_before_'.format(i), ig)
            image, joints = self.dataset.apply_translate(image, joints)
            self.draw_joints(image, joints, 'trans_{}_after_'.format(i), ig)

    def test_apply_rotate(self):
        for i, (img_id, joints) in enumerate(self.dataset.joints):
            image = self.dataset.images[img_id]
            ig, bbox_w, bbox_h, center_x, center_y = self.dataset.info[i]
            self.draw_joints(image, joints, 'rotate_{}_before_'.format(i), ig)
            image, joints = self.dataset.apply_rotate(image, joints, ig)
            self.draw_joints(image, joints, 'rotate_{}_after_'.format(i), ig)

    def test_apply_coord_normalize(self):
        for image_id, joints in self.dataset.joints:
            image = self.dataset.images[image_id]
            image, joints = self.dataset.apply_coord_normalize(image, joints)

    def test_apply_gcn(self):
        for image_id, joints in self.dataset.joints:
            image = self.dataset.images[image_id]
            image, joints = self.dataset.apply_gcn(image, joints)
            np.testing.assert_allclose(
                image.reshape(-1, 3).mean(axis=0), [0, 0, 0], atol=1e-5)
            np.testing.assert_allclose(
                image.reshape(-1, 3).std(axis=0), [1., 1., 1.], atol=1e-5)

    def test_flic(self):
        img_dir = 'data/FLIC-full/images'
        symmetric_joints = '[[2, 4], [1, 5], [0, 6]]'
        np.random.rand(3)
        dataset = self.create_dataset(
            self.flic_csv,
            img_dir=img_dir,
            symmetric_joints=symmetric_joints,
            fliplr=True,
            rotate=True,
            rotate_range=10,
            zoom=True,
            base_zoom=1.5,
            zoom_range=0.2,
            translate=True,
            translate_range=5,
            coord_normalize=False,
            gcn=False,
        )
        self.assertEqual(len(dataset), self.n_test)
        for i in range(len(dataset)):
            image, joints, ignore_joints = dataset.get_example(i)
            image = image.astype(np.uint8)
            self.draw_joints(
                image, joints, 'flic_{}_'.format(i), ignore_joints)

    def test_lsp(self):
        img_dir = 'data/lspet_dataset/images'
        symmetric_joints = '[[8, 9], [7, 10], [6, 11], [2, 3], [1, 4], [0, 5]]'
        np.random.rand(3)
        dataset = self.create_dataset(
            self.lsp_csv,
            img_dir=img_dir,
            symmetric_joints=symmetric_joints,
            fliplr=True,
            rotate=True,
            rotate_range=10,
            zoom=True,
            base_zoom=1.5,
            zoom_range=0.2,
            translate=True,
            translate_range=5,
            coord_normalize=False,
            gcn=False,
        )
        self.assertEqual(len(dataset), self.n_test)
        for i in range(len(dataset)):
            image, joints, ignore_joints = dataset.get_example(i)
            image = image.astype(np.uint8)
            self.draw_joints(
                image, joints, 'lsp_{}_'.format(i), ignore_joints)

    def test_mpii(self):
        img_dir = 'data/mpii/images'
        symmetric_joints = \
            '[[12, 13], [11, 14], [10, 15], [2, 3], [1, 4], [0, 5]]'
        np.random.rand(3)
        dataset = self.create_dataset(
            self.mpii_csv,
            img_dir=img_dir,
            symmetric_joints=symmetric_joints,
            fliplr=True,
            rotate=True,
            rotate_range=10,
            zoom=True,
            base_zoom=1.5,
            zoom_range=0.2,
            translate=True,
            translate_range=5,
            coord_normalize=False,
            gcn=False,
        )
        self.assertEqual(len(dataset), self.n_test)
        for i in range(len(dataset)):
            image, joints, ignore_joints = dataset.get_example(i)
            image = image.astype(np.uint8)
            self.draw_joints(
                image, joints, 'mpii_{}_'.format(i), ignore_joints)
