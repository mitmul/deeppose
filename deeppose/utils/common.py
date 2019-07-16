import numpy as np
from chainercv import transforms


def crop_with_joints(img, point, scale_h=1.5, sacle_w=1.2):
    min_y, min_x = point.min(axis=0)
    max_y, max_x = point.max(axis=0)

    width = max_x - min_x
    height = max_y - min_y

    new_width = width * sacle_w
    new_height = height * scale_h

    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2

    _, img_height, img_width = img.shape
    new_min_x = int(np.clip(center_x - new_width / 2, 0, img_width))
    new_max_x = int(np.clip(new_min_x + new_width, 0, img_width))
    new_min_y = int(np.clip(center_y - new_height / 2, 0, img_height))
    new_max_y = int(np.clip(new_min_y + new_height, 0, img_width))

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
