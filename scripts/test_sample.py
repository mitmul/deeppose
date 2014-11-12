#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import caffe
import cv2 as cv
import numpy as np
from skvideo.io import VideoCapture, VideoWriter


def draw_joints(img, net):
    img = cv.resize(img, (227, 227))
    img = img.swapaxes(0, 2).swapaxes(1, 2)
    net.blobs['data'].data[:, :, :, :] = img
    joints = net.forward().values()[0].flatten()
    joints = np.asarray(joints)
    img = img.swapaxes(0, 2).swapaxes(0, 1)

    joints = joints.reshape((7, 2))

    for j, joint in enumerate(joints):
        if j != 2 and j != 3 and j + 1 < len(joints):
            sj = (int(joints[j, 0] * 227),
                  int(joints[j, 1] * 227))
            nj = (int(joints[j + 1, 0] * 227),
                  int(joints[j + 1, 1] * 227))
            cv.line(img, sj, nj, (0, 255, 0), 3)

    sj = (int(joints[2, 0] * 227),
          int(joints[2, 1] * 227))
    nj = (int(joints[4, 0] * 227),
          int(joints[4, 1] * 227))
    cv.line(img, sj, nj, (0, 255, 0), 3)

    sj = ((sj[0] + nj[0]) / 2,
          (sj[1] + nj[1]) / 2)
    nj = (int(joints[3, 0] * 227),
          int(joints[3, 1] * 227))
    cv.line(img, sj, nj, (0, 255, 0), 3)

    for j, joint in enumerate(joints):
        joint = (int(joint[0] * 227), int(joint[1] * 227))
        cv.circle(img, joint, 5, (0, 0, 255), -1)

    sj = (int(joints[2, 0] * 227),
          int(joints[2, 1] * 227))
    nj = (int(joints[4, 0] * 227),
          int(joints[4, 1] * 227))
    cv.circle(img, ((sj[0] + nj[0]) / 2,
                    (sj[1] + nj[1]) / 2),
              5, (0, 0, 255), -1)

    return img


def test_on_video(net, vid_fn, out_fn):
    # capture
    cap = VideoCapture(vid_fn)
    ret, img = cap.read()

    # writer
    orig_shape = (227, 227)  # img.shape
    wri = VideoWriter(out_fn, frameSize=(orig_shape[1], orig_shape[0]))
    wri.open()

    frame = 0
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            img = draw_joints(img, net)
            # img = cv.resize(img, (orig_shape[1], orig_shape[0]))
            wri.write(img)
            frame += 1
            print frame
        else:
            print 'finished'
    cap.release()
    wri.release()


def test_on_images(net, img_dir, out_dir, num=-1, ext='.jpg'):
    print '%s/*%s' % (img_dir, ext)
    for i, img_fn in enumerate(glob.glob('%s/*%s' % (img_dir, ext))):
        img = cv.imread(img_fn)
        img = draw_joints(img, net)
        cv.imwrite('%s/%s' % (out_dir, os.path.basename(img_fn)), img)
        print img_fn

        if num != -1 and i > num:
            break


def load_net(net_dir):
    _, fn = sorted([(int(fn.split('_')[-1].split('.')[0]), fn)
                    for fn in glob.glob(
                        '%s/snapshots/*.caffemodel' % net_dir)])[-1]
    net = caffe.Net('%s/predict.prototxt' % net_dir, fn)
    net.set_mode_gpu()
    print fn

    return net

if __name__ == '__main__':
    net_dir = 'results/AlexNet_2014-11-12_09-59-14'
    net = load_net(net_dir)
    if not os.path.exists('data/test'):
        os.mkdir('data/test')
    # test_on_images(net, 'data/FLIC-full/crop', 'FLIC-full', 100)
    test_on_video(net, 'data/test/yahagi_crop1.mov',
                  'data/test/yahagi_joint1.avi')
    test_on_video(net, 'data/test/ogi_crop1.mov',
                  'data/test/ogi_joint1.avi')
