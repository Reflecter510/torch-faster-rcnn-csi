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

# 动作框与分类 转换为 逐帧预测的分类结果
def locCls2Label(location, data_class):
    labels = []
    for i in range(0, location.shape[0]):
        st = int(location[i][0])
        ed = int(location[i][1])
        head = torch.zeros(st)
        action_label = torch.zeros(ed-st)
        action_label[:] = data_class[i]
        tail = torch.zeros(192-ed)
        label = torch.cat([head, action_label, tail])
        labels.append(label.view(1,-1))
    return torch.cat(labels).tolist()


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes, savename, title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=250)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if x_val == y_val:
            color = "white"
        else:
            color = "black"
        if len(classes) < 7:
            fontsize=18
        else:
            fontsize=9
        if c > 0.001:
            plt.text(x_val, y_val, "%0.1f%%" % ((c*100),), color=color, fontsize=fontsize, va='center', ha='center')
        else:
            plt.text(x_val, y_val, "%0.1f%%" % ((c*100),), color=color, fontsize=fontsize, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=45)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()