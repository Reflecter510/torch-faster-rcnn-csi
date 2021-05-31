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
    #plt.show()
    plt.close()

def draw_bar(labels,quants, methods, title):
    # 这两行代码解决 plt 中文显示的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.ylim((min(1.0, np.min(quants)-0.04), np.max(quants)+0.01))
    
    bar_width = 0.2 # 条形宽度
    index_0 = np.arange(len(labels)) # 第一个条形图的横坐标
    index = index_0
    colors = ["royalblue", "darkorange" , "slategrey", "gold", "red"]
    # 使用两次 bar 函数画出两组条形图
    for i in range(0,quants.shape[0]):
        index = index_0 if i==0 else index+bar_width
        plt.bar(index, height=quants[i], width=bar_width, color=colors[i], label=methods[i])
    
    plt.legend() # 显示图例
    plt.xticks(index_0 + bar_width/2, labels) # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    plt.ylabel('') # 纵坐标轴标题
    plt.title(title) # 图形标题
    
    plt.show()

def draw_table(data):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    table = plt.table(cellText=data, colLabels=['逐帧分类精度', '逐帧检测精度', '逐帧分类精度', '逐帧检测精度','逐帧分类精度','逐帧检测精度'], loc='center', 
                cellLoc='center',)# colColours=['#FFFFFF', '#F3CC32', '#2769BD', '#DC3735'])
    table.auto_set_font_size(False)
    h = table.get_celld()[(0,0)].get_height()
    w = table.get_celld()[(0,0)].get_width()
    header = [table.add_cell(-1,pos, w, h, loc="center", facecolor="none") for pos in [0,1,2,3,4,5]]
    header[0].visible_edges = "TBL"
    header[1].visible_edges = "TBR"
    header[2].visible_edges = "TBL"
    header[3].visible_edges = "TBR"
    header[4].visible_edges = "TBL"
    header[5].visible_edges = "TBR"
    header[0].get_text().set_text("S1")
    header[2].get_text().set_text("S2")
    header[4].get_text().set_text("TEMPORAL")

    plt.axis('off')
    plt.show()
    plt.close()
