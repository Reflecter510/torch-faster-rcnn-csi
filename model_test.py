import os
from nets.model import get_model
from dataset import S1P1, S2, TEMPORAL
import torch
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from utils.utils import detection_acc, bbox_iou, locCls2Label, plot_confusion_matrix
from utils import DataUtil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
       
os.system("rm predict/*.png")

# 数据集设置:  S1P1 或 S2 或 TEMPORAL
dataset = TEMPORAL
# 训练集"train" 或 测试集 "test"
which_data = "test"

# 测试集是否添加噪声
DataUtil.noise = False

'''模型断点路径'''
path_checkpoint =  "结果/TEMPORAL/exp0-vgg/Epoch220-Total_Loss0.0155-Val_Loss0.3717.pth"

#结果可视化
PLOT = False    

# 主干特征提取网络
BACKBONE = "vgg"

#--------------------------------------------------------------------------------------------
# 根据数据集初始化相关变量
dataset_name = dataset.name
NUM_CLASSES = dataset.num_classes
test_bacth = dataset.test_batch
ANCHOR = dataset.anchor
IMAGE_SHAPE = dataset.image_shape
actions = dataset.actions

#加载测试集
num_test_instances, test_data_loader = DataUtil.get_data_loader(dataset_name, which_data, batch_size=test_bacth, shuffle = False)

#------------------------------  模型  ----------------------------------------#
# 是否支持Cuba
Cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if Cuda else "cpu")
print("use" , device)

# Faster RCNN
model = get_model(dataset, BACKBONE).to(device)
#-----------------------------------------------------------------------------#

# 加载断点
checkpoint = torch.load(path_checkpoint,  map_location=device)  
# 加载模型可学习参数
model.load_state_dict(checkpoint['net'])  
# 测试模式
model = model.eval()
#-----------------------------------------------------------------------------#


# 初始化结果
np.set_printoptions(suppress=True)
ious_all = 0
detection_all = 0.0
final_all = 0.0
acc = 0.0
cnt = 0
i = 0
_time = 0.0
# 逐帧预测的标签列表
pred_labels = []
gt_labels = []

for (data, bbox, label) in tqdm(test_data_loader):
    with torch.no_grad():
        data = data.reshape([-1,IMAGE_SHAPE[0], IMAGE_SHAPE[1]])
        if Cuda:
            dataV = Variable(data).cuda()
            bboxV = Variable(bbox).cuda()
            labelV = Variable(label).cuda()
        else:
            dataV = Variable(data)
            bboxV = Variable(bbox)
            labelV = Variable(label)

        # 注意，测试时间使用cpu，gpu测试时需要使用cuda同步才准
        start = time.time()
        # 预测
        predictions = model(dataV.unsqueeze(3))
        end = time.time()

        _time += (end-start)

        # 对批量预测进行结果统计
        for idp in range(0, len(predictions[1])):
            prediction = predictions[1][idp]
            i+=1
            if prediction['boxes'].shape[0] == 0:
                continue
            # 获取预测动作框
            bbox = prediction["boxes"][:,1:4:2].view(-1,2)
            conf = prediction["scores"]
            label = prediction["labels"]
            idx = 0

            # 计算IoU、逐帧检测精度、逐帧分类精度
            max_iou = bbox_iou(np.asarray(bbox[idx].view(1,2).cpu()), np.asarray(bboxV[idp].view(1,2)))[0][0]
            dete_acc = detection_acc(np.asarray(bbox[idx].view(1,2).cpu()), np.asarray(bboxV[idp].view(1,2)))[0][0]
            final_acc = detection_acc(np.asarray(bbox[idx].view(1,2).cpu()), np.asarray(bboxV[idp].view(1,2)), ACC=int(label[idx])==int(labelV[idp][0]))[0][0]

            # 动作框转换为逐帧预测的标签
            gt_labels.extend(locCls2Label(bboxV[idp].view(-1,2), labelV[idp][0].view(-1,1))[0])
            pred_labels.extend(locCls2Label(bbox[idx].view(-1,2).cpu(), label[idx].view(-1,1))[0])

            # 绘制动作实例图
            if PLOT:
                X=np.linspace(0,192,192,endpoint=True)
                # 绘制CSI热力图
                plt.contourf(dataV[idp].cpu())
                plt.colorbar()

                currentAxis=plt.gca()
                if dataset_name=="TEMPORAL":
                    h = 49
                else:
                    h = 87
                # 绘制真实动作框
                rect=patches.Rectangle((bboxV[idp][0], 1), bboxV[idp][1]-bboxV[idp][0] ,h, linewidth=2,edgecolor='white',facecolor='none')
                currentAxis.add_patch(rect)
                # 绘制预测动作框
                rect=patches.Rectangle((bbox[idx][0], 2), bbox[idx][1]-bbox[idx][0] ,h-2, linewidth=2,edgecolor='r',facecolor='none')
                currentAxis.add_patch(rect)
                plt.ylabel("Channels")
                plt.xlabel("predict_label= "+actions[int(label[idx])]+"  groudtruth_label = "+actions[int(labelV[idp][0])]+"  score = "+str(conf[idx].tolist())[:5])

                # plt.xlabel(which_data+str(i)+"  prd = "+actions[int(label[idx])]+"  gt = "+actions[int(labelV[idp][0])]+"  acc = "+str(conf[idx].tolist())[:5]+"  iou = "+str(max_iou)[:5]+" wd = "+str(int(bboxV[idp][1]-bboxV[idp][0])))
                plt.savefig("predict/%d.png"%(i), dpi=128)
                plt.close()
          
            #print("预测:", bbox[idx].tolist()," ", str(conf[idx].tolist())[:5], " ",label[idx], " 真实:",bboxV[idp].tolist(), labelV[idp].tolist(), "iou=", max_iou)
            
            # 累加结果
            if int(label[idx])==int(labelV[idp][0]):
                acc = acc + 1
            ious_all += max_iou
            detection_all += dete_acc
            final_all += final_acc

            cnt+=1

# 绘制逐帧分类混淆矩阵
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(gt_labels, pred_labels, normalize="true")
plot_confusion_matrix(matrix, dataset.actions,'confusion_matrix.png', title='Sample-level Action Classification Confusion Matrix')
# 绘制逐帧检测混淆矩阵
gt_labels = [1.0 if each>=1.0 else 0.0 for each in gt_labels]
pred_labels = [1.0 if each>=1.0 else 0.0 for each in pred_labels]
matrix = confusion_matrix(gt_labels, pred_labels, normalize="true")
plot_confusion_matrix(matrix, ["non","yes"],'detection_confusion_matrix.png', title='Sample-level Action Detection Confusion Matrix')

# 计算平均结果
ious_all /= i
detection_all /= i
final_all /= i
acc /= i
_time /= i
print("有效预测：",cnt)
print("IOU: ",ious_all)
print("分类准确度：", acc)
print("检测精度: ",detection_all)
print("检测分类精度: ",final_all)
print("平均每份预测时间：", _time,"秒")