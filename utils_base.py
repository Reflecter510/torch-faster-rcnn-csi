from nets.alexnet import get_alex_feat_stride
#from nets.CNN1d import get_cnn1d_feat_stride

import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader

home_dir = './'

def get_IMAGE_SHAPE_from_dataset_name(dataset_name):
    IMAGE_SHAPE = []
    if dataset_name == "192S1ALL" or dataset_name == "192S2":
        IMAGE_SHAPE = [90,192,1]
    elif dataset_name == "TEMPORAL":
        IMAGE_SHAPE = [52,192,1]
    elif dataset_name == 'S1O1':
        IMAGE_SHAPE = [90,1500,1]
    return IMAGE_SHAPE

# alexnet 根据数据尺寸获取步长
def get_feat_stride(x, backbone="alexnet"):
    if backbone=="vgg":
        if x==192:
            return 192.0/12
        if x==1500:
            return 1599.0/125
    if backbone=="alexnet":
        return get_alex_feat_stride(x)
    # elif backbone=="cnn1d":
    #     return get_cnn1d_feat_stride(x)
    return 0


# 计算有效anchor框与目标框的IOU  IOU:一般指代模型预测的 bbox 和 Groud Truth 之间的交并比
def compute_iou(valid_anchor_boxes, bbox):
    valid_anchor_num = len(valid_anchor_boxes)
    ious = np.zeros((valid_anchor_num, 2), dtype=np.float32)
    for num1, i in enumerate(valid_anchor_boxes):
        xa1,xa2 = i
        anchor_area = xa2 - xa1  # anchor框面积
        for num2, j in enumerate(bbox):
            xb1, xb2 = j
            box_area = xb2 - xb1  # 目标框面积
            inter_x1 = max([xb1, xa1])
            inter_x2 = min([xb2, xa2])
            if inter_x1 < inter_x2:
                iter_area = inter_x2 - inter_x1  # anchor框和目标框的相交面积
                iou = iter_area / (anchor_area + box_area - iter_area)  # IOU计算
            else:
                iou = 0.
            ious[num1, num2] = iou
    return ious




if __name__ == "__main__":
    # a = init_anchor()
    # bboxV = np.asarray([[20, 500], [300, 600]], dtype=np.float32)
    # labelV = np.asarray([6, 8], dtype=np.int8)
    # img_tensor = torch.zeros((1, 1, 90, 1500)).float()
    # dataV = torch.autograd.Variable(img_tensor)
    # loss = compute_loss(bboxV,labelV,dataV)

    print(get_alex_feat_stride(90*192))
    pass
     
    