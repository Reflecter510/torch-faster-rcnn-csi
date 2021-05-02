import torch

import numpy as np
from torch.nn import functional as F

def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 2 or bbox_b.shape[1] != 2:
        print(bbox_a, bbox_b)
        raise IndexError
    # top left
    tl = np.maximum(bbox_a[:, None, :1], bbox_b[:, :1])
    # bottom right
    br = np.minimum(bbox_a[:, None, 1:], bbox_b[:, 1:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 1:] - bbox_a[:, :1], axis=1)
    area_b = np.prod(bbox_b[:, 1:] - bbox_b[:, :1], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def detection_acc(bbox_a, bbox_b, ACC=True):
    if bbox_a.shape[1] != 2 or bbox_b.shape[1] != 2:
        print(bbox_a, bbox_b)
        raise IndexError
    # top left
    tl = np.maximum(bbox_a[:, None, :1], bbox_b[:, :1])
    # bottom right
    br = np.minimum(bbox_a[:, None, 1:], bbox_b[:, 1:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 1:] - bbox_a[:, :1], axis=1)
    area_b = np.prod(bbox_b[:, 1:] - bbox_b[:, :1], axis=1)

    if ACC:
        return 1.0 - (area_a[:, None] + area_b - 2 * area_i) / 192.0
    else:
        return 1.0 - (area_a[:, None] + area_b - area_i) / 192.0

