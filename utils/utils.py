from matplotlib import patches
import torch
import numpy as np


# 计算IoU
def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 2 or bbox_b.shape[1] != 2:
        print(bbox_a, bbox_b)
        raise IndexError
    # top left
    tl = np.maximum(bbox_a[:, None, :1], bbox_b[:, :1])
    # bottom right
    br = np.minimum(bbox_a[:, None, 1:], bbox_b[:, 1:])
    # 相交区间长度
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    # 动作框a长度
    area_a = np.prod(bbox_a[:, 1:] - bbox_a[:, :1], axis=1)
    # 动作框b长度
    area_b = np.prod(bbox_b[:, 1:] - bbox_b[:, :1], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

# 计算逐帧检测精度或逐帧分类精度（序列长度固定为192）
def detection_acc(bbox_a, bbox_b, ACC=True):
    if bbox_a.shape[1] != 2 or bbox_b.shape[1] != 2:
        print(bbox_a, bbox_b)
        raise IndexError
    # top left
    tl = np.maximum(bbox_a[:, None, :1], bbox_b[:, :1])
    # bottom right
    br = np.minimum(bbox_a[:, None, 1:], bbox_b[:, 1:])
    # 相交区间长度
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    # 动作框a长度
    area_a = np.prod(bbox_a[:, 1:] - bbox_a[:, :1], axis=1)
    # 动作框b长度
    area_b = np.prod(bbox_b[:, 1:] - bbox_b[:, :1], axis=1)

    if ACC:
        return 1.0 - (area_a[:, None] + area_b - 2 * area_i) / 192.0
    else:
        return 1.0 - (area_a[:, None] + area_b - area_i) / 192.0

#---------------------------------------------------------------------
# 下列代码用于summary_test.py

# 动作框与分类 转换为 逐帧预测的分类结果
def locCls2Label(location, data_class):
    labels = []
    for i in range(0, location.shape[0]):
        # 动作起始和结束下标
        st = int(location[i][0])
        ed = int(location[i][1])
        # 动作前景
        head = torch.zeros(st)
        # 动作区间
        action_label = torch.zeros(ed-st)
        action_label[:] = data_class[i]
        # 动作后景
        tail = torch.zeros(192-ed)
        # 拼接三个部分的结果
        label = torch.cat([head, action_label, tail])
        labels.append(label.view(1,-1))
    return torch.cat(labels).tolist()

# 逐帧预测的分类结果 转换为 动作框与分类
def label2LocCls(prediction):
    predictions = [{},[]]
    # 逐帧结果转换为动作框计算IoU和分类准确度
    for j in range(0, prediction.shape[0]):
        # 计算预测结果不为0的下标
        pred_index = prediction[j].nonzero()
        if pred_index.shape[0] == 0:
            continue
        # 动作框分别取下标的最小值和最大值
        pred_box = torch.Tensor([[pred_index[0], pred_index[-1]]])
        # 动作类别标签直接取为动作框中点的标签
        pred_action = prediction[j][int(pred_box.mean())]
        # 按照 [ {}, [{动作框字典},...,{}] ]  格式生成预测结果
        predictions[1].append({"boxes":torch.Tensor([[0,pred_box[0][0],1,pred_box[0][1]]]), "scores":torch.Tensor([0.0]), "labels":pred_action.view(1,1)})
    return predictions

# 绘制动作实例图
def plot_csi_box_actions(data, pred_box, gt_box, pred_action, gt_action,score, dataset_name, i):
    X=np.linspace(0,192,192,endpoint=True)
    # 绘制CSI热力图
    plt.contourf(data)
    plt.colorbar()

    currentAxis=plt.gca()
    if dataset_name=="TEMPORAL":
        h = 49
    else:
        h = 87
    # 绘制真实动作框
    rect=patches.Rectangle((gt_box[0], 1), gt_box[1]-gt_box[0] ,h, linewidth=2,edgecolor='white',facecolor='none')
    currentAxis.add_patch(rect)
    # 绘制预测动作框
    rect=patches.Rectangle((pred_box[0], 2), pred_box[1]-pred_box[0] ,h-2, linewidth=2,edgecolor='r',facecolor='none')
    currentAxis.add_patch(rect)
    plt.ylabel("Channels")
    plt.xlabel("predict_label= "+pred_action+"  groudtruth_label = "+gt_action+"  score = "+score)
    plt.savefig("predict/%d.png"%(i), dpi=98)
    plt.close()


# 绘制混淆矩阵
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
def plot_confusion_matrix(cm, classes, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=115)
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
            fontsize=24
        else:
            fontsize=12
 
        plt.text(x_val, y_val, "%0.1f%%" % ((c*100),), color=color, fontsize=fontsize, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=45,fontsize=fontsize)
    plt.yticks(xlocations, classes,fontsize=fontsize)
    plt.ylabel('Actual label',fontsize=fontsize)
    plt.xlabel('Predict label',fontsize=fontsize)
    
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
    plt.close()

# 绘制柱状图
def draw_bar(labels,quants, methods, title):
    # 这两行代码解决 plt 中文显示的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    if methods[0] == "滑动窗口" and title=="分类准确度":
        plt.ylim((min(1.0, np.min(quants[1:])-0.04), np.max(quants)+0.01))
    else:
        plt.ylim((min(1.0, np.min(quants)-0.04), np.max(quants)+0.01))
    
    bar_width = 0.2 # 条形宽度
    index_0 = np.arange(len(labels)) # 第一个条形图的横坐标
    index = index_0
    colors = ["royalblue", "darkorange" , "slategrey", "gold", "red"]
    # 使用两次 bar 函数画出两组条形图
    for i in range(0,quants.shape[0]):
        index = index_0 if i==0 else index+bar_width
        if methods[i] == "滑动窗口" and title=="分类准确度":
            continue
        plt.bar(index, height=quants[i], width=bar_width, color=colors[i], label=methods[i])
    
    plt.legend() # 显示图例
    plt.xticks(index_0 + bar_width/2, labels) # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    plt.ylabel('') # 纵坐标轴标题
    plt.title(title) # 图形标题
    
    plt.show()

# 绘制表格
# input: ndarrray
def draw_table(data, methods):
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

    header = [table.add_cell(pos,-1, w, h, loc="center", facecolor="none") for pos in range(-1, len(methods)+1)]
    header[0].get_text().set_text("测试集")
    for i in range(2, len(methods)+2):
        header[i].get_text().set_text(methods[i-2])

    plt.axis('off')
    plt.show()
    plt.close()

# 基于阈值的滑动窗口方法
def slide_window(data):
    predictions = [{},[]]
    for i in range(0, data.shape[0]):   #batch
        # 幅值归一化
        data[i] = (data[i]-torch.min(data[i]))/(torch.max(data[i])-torch.min(data[i]))
        # 差分
        diff_amp = torch.diff(data[i], dim=0)
        # 窗口大小
        window = 35
        # 阈值
        threshold = torch.std(torch.abs(diff_amp))
        
        begin = []
        s = []
        e = []
        # 窗口滑动
        for slide in range(1, diff_amp.shape[0]-window):
            tmp_amp = torch.mean(torch.abs(diff_amp[slide:slide+window-1]))
            # 检测动作起始点
            if begin == [] and tmp_amp > 1.18*threshold:
                begin = slide
                s.append(begin)
            # 检测动作结束点
            if begin != [] and tmp_amp < 1.39*threshold:
                over = slide
                e.append(over)
        
        if s == [] or e == []:
            s = [0.0]
            e = [0.0]

        # 直接取第一个起始点和最后一个结束点作为动作框
        pred_box = torch.Tensor([[0,s[0],1,e[-1]]])
        
        # label 和 scores 全部设为默认值 0.0 
        predictions[1].append({"boxes":pred_box, "scores":torch.Tensor([0.0]), "labels":torch.Tensor([0])})
        
        # 绘制CSI差分图以及动作框
        # if 0:
        #     segment = torch.mean(diff_amp.abs(), 1)
        #     plt.plot(range(0,segment.shape[0]), segment)
        #     plt.xlabel("Times")
        #     plt.ylabel("Amp_difference")
        #     h = torch.max(segment[:-1])-torch.min(segment[:-1])
        #     bottom = torch.min(segment[:-1])
        #     currentAxis=plt.gca()
        #     rect=patches.Rectangle((bboxV[i][0], bottom-0.001), bboxV[i][1]-bboxV[i][0] ,h+0.002, linewidth=2,edgecolor='green',facecolor='none')
        #     currentAxis.add_patch(rect)
        #     rect=patches.Rectangle((s[0], bottom), e[-1]-s[0] ,h, linewidth=2,edgecolor='red',facecolor='none')
        #     currentAxis.add_patch(rect)
        #     plt.show()
    return predictions