import os
from dataset import S1P1, S2,TEMPORAL
import torch
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from utils.utils import detection_acc, bbox_iou
from utils import DataUtil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
       
os.system("rm predict/*.png")

# 数据集设置:  S1P1 或 TEMPORAL
dataset = S2
which_data = "test"

#结果可视化
PLOT = False    

#--------------------------------------------------------------------------------------------
dataset_name = dataset.name
NUM_CLASSES = dataset.num_classes
test_bacth = dataset.test_batch
ANCHOR = dataset.anchor
IMAGE_SHAPE = dataset.image_shape
actions = dataset.actions
#加载测试集
num_test_instances, test_data_loader = DataUtil.get_data_loader(dataset_name, which_data, batch_size=test_bacth, shuffle = False)

Cuda = torch.cuda.is_available()

'''
input: CSI data with shape(batch, times_len, channels)
output: 
'''
def slide_window(data, bbox):
    predictions = [{},[]]
    #TODO 差分  滑窗
    for i in range(0, data.shape[0]):   #batch
        # 幅值归一化
        data[i] = (data[i]-torch.min(data[i]))/(torch.max(data[i])-torch.min(data[i]))
        # 差分
        diff_amp = torch.diff(data[i], dim=0)
        
        window = 20
        threshold = torch.std(torch.abs(diff_amp))
        begin = []
        s = []
        e = []
        for slide in range(1, diff_amp.shape[0]-window):
            tmp_amp = torch.mean(torch.abs(diff_amp[slide:slide+window-1]))
            if begin == [] and tmp_amp > 0.45*threshold:
                begin = slide
                s.append(begin)
            if begin != [] and tmp_amp < threshold:
                over = slide
                e.append(over)
        
        if s == [] and e == []:
            s.append(0)
            e.append(diff_amp.shape[0])
   
        predictions[1].append({"boxes":torch.Tensor([[0,s[0],1,e[-1]]]), "scores":torch.Tensor([0.0]), "labels":torch.Tensor([0])})
        
        if 1:
            segment = torch.mean(diff_amp.abs(), 1)
            plt.plot(range(0,segment.shape[0]), segment)
            plt.xlabel("Times")
            plt.ylabel("Amp_difference")
            h = torch.max(segment[:-1])-torch.min(segment[:-1])
            bottom = torch.min(segment[:-1])
            currentAxis=plt.gca()
            rect=patches.Rectangle((bboxV[i][0], bottom-0.001), bboxV[i][1]-bboxV[i][0] ,h+0.002, linewidth=2,edgecolor='green',facecolor='none')
            currentAxis.add_patch(rect)
            rect=patches.Rectangle((s[0], bottom), e[-1]-s[0] ,h, linewidth=2,edgecolor='red',facecolor='none')
            currentAxis.add_patch(rect)
            plt.show()
    return predictions


# 初始化结果
np.set_printoptions(suppress=True)
ious_all = 0
detection_all = 0.0
final_all = 0.0
acc = 0.0
cnt = 0
i = 0
_time = 0.0

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

        start = time.time()
        predictions = slide_window(dataV.permute(0,2,1), bboxV)
        end = time.time()

        _time += (end-start)

        for idp in range(0, len(predictions[1])):
            prediction = predictions[1][idp]
            i+=1
            if prediction['boxes'].shape[0] == 0:
                continue
           
            bbox = prediction["boxes"][:,1:4:2].view(-1,2)
            conf = prediction["scores"]
            label = prediction["labels"]
            idx = 0

            max_iou = bbox_iou(np.asarray(bbox[idx].view(1,2).cpu()), np.asarray(bboxV[idp].view(1,2)))[0][0]
            dete_acc = detection_acc(np.asarray(bbox[idx].view(1,2).cpu()), np.asarray(bboxV[idp].view(1,2)))[0][0]
            final_acc = detection_acc(np.asarray(bbox[idx].view(1,2).cpu()), np.asarray(bboxV[idp].view(1,2)), ACC=int(label[idx])==int(labelV[idp][0]))[0][0]

            if PLOT:
                X=np.linspace(0,192,192,endpoint=True)
                plt.contourf(dataV[idp].cpu())
                plt.colorbar()

                currentAxis=plt.gca()
                if dataset_name=="TEMPORAL":
                    h = 49
                else:
                    h = 87
                rect=patches.Rectangle((bboxV[idp][0], 1), bboxV[idp][1]-bboxV[idp][0] ,h, linewidth=2,edgecolor='white',facecolor='none')
                currentAxis.add_patch(rect)
                rect=patches.Rectangle((bbox[idx][0], 2), bbox[idx][1]-bbox[idx][0] ,h-2, linewidth=2,edgecolor='r',facecolor='none')
                currentAxis.add_patch(rect)
                plt.ylabel("Channels")
                plt.xlabel("predict_label= "+actions[int(label[idx])]+"  groudtruth_label = "+actions[int(labelV[idp][0])]+"  score = "+str(conf[idx].tolist())[:5])

                # plt.xlabel(which_data+str(i)+"  prd = "+actions[int(label[idx])]+"  gt = "+actions[int(labelV[idp][0])]+"  acc = "+str(conf[idx].tolist())[:5]+"  iou = "+str(max_iou)[:5]+" wd = "+str(int(bboxV[idp][1]-bboxV[idp][0])))
                plt.savefig("predict/%d.png"%(i), dpi=128)
                plt.close()
          
            #print("预测:", bbox[idx].tolist()," ", str(conf[idx].tolist())[:5], " ",label[idx], " 真实:",bboxV[idp].tolist(), labelV[idp].tolist(), "iou=", max_iou)
            
            if int(label[idx])==int(labelV[idp][0]):
                acc = acc + 1
            ious_all += max_iou
            detection_all += dete_acc
            final_all += final_acc

            cnt+=1

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