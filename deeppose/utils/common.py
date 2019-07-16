import math

import numpy as np
from chainercv import transforms


def crop_with_joints(img, point, scale_h=1.5, sacle_w=1.2, random_offset_ratio_y=0, random_offset_ratio_x=0):
    min_y, min_x = point.min(axis=0)
    max_y, max_x = point.max(axis=0)
    _, img_height, img_width = img.shape

    # Zero-padding
    if min_y < 0:
        np.pad(img, ((0, 0), (math.ceil(-min_y), 0), (0, 0)), 'constant')
        min_y = 0
    if min_x < 0:
        np.pad(img, ((0, 0), (0, 0), (math.ceil(-min_x), 0)), 'constant')
        min_x = 0
    if max_y > img_height:
        np.pad(img, ((0, 0), (0, math.ceil(max_y - img_height)), (0, 0)), 'constant')
        max_y = img_height - 1
    if max_x > img_width:
        np.pad(img, ((0, 0), (0, 0), (0, math.ceil(max_x - img_width))), 'constant')
        max_x = img_width - 1

    width = max_x - min_x
    height = max_y - min_y

    new_width = width * sacle_w
    new_height = height * scale_h

    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2

    new_min_x = int(np.clip(center_x - new_width / 2, 0, img_width))
    new_max_x = int(np.clip(new_min_x + new_width, 0, img_width))
    new_min_y = int(np.clip(center_y - new_height / 2, 0, img_height))
    new_max_y = int(np.clip(new_min_y + new_height, 0, img_width))

    offset_y = random_offset_ratio_y * new_height / 2
    offset_y = np.random.uniform(-offset_y, offset_y)
    offset_x = random_offset_ratio_x * new_width / 2
    offset_x = np.random.uniform(-offset_x, offset_x)
    new_min_x = int(np.clip(new_min_x + offset_x, 0, min_x))
    new_max_x = int(np.clip(new_max_x + offset_x, max_x, img_width))
    new_min_y = int(np.clip(new_min_y + offset_y, 0, min_y))
    new_max_y = int(np.clip(new_max_y + offset_y, max_y, img_height))

    crop = img[:, new_min_y:new_max_y, new_min_x:new_max_x]
    point = point - np.array([new_min_y, new_min_x])

    return crop, point


def to_square(img, point, size=(220, 220)):
    in_size = img.shape[1:]  # (H, W)
    img = transforms.resize(img, size)
    point = transforms.resize_point([point], in_size, size)

    return img, point[0]


def lr_flip(img, point):
    _, height, width = img.shape
    img = transforms.flip(img, x_flip=True)
    point = transforms.flip_point([point], (height, width), x_flip=True)[0]

    return img, point
