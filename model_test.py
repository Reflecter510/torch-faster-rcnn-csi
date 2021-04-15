from functools import update_wrapper
import utils_base
import torch
from tqdm import tqdm, utils
from torch.autograd import Variable
import numpy as np
from utils.utils import detection_acc, loc2bbox, nms, DecodeBox, bbox_iou, temporal_iou
import os
from utils import DataUtil

from lib.pool import MultiScaleRoIAlign
from lib.Faster_RCNN import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from nets.alexnet import AlexNet

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import time
       
os.system("rm input/detection-results/result*.txt")
os.system("rm input/ground-truth/result*.txt")
os.system("rm predict/*.png")

dataset_name = "192S1ALL"
which_data = "test"
IMAGE_SHAPE = utils_base.get_IMAGE_SHAPE_from_dataset_name(dataset_name)

if dataset_name == "TEMPORAL":
    actions = ['none', "1", "2", "3", "4", "5", "6"]
else:
    actions = ['none', 'jump', 'pick', 'throw', 'pull', 'clap', 'box', 'wave', 'lift', 'kick', 'squat', 'turnRound', 'checkWatch']

'''
加载测试集
'''
if dataset_name != "S":
    num_test_instances, test_data_loader = DataUtil.get_data_loader(dataset_name, which_data, batch_size=1, shuffle = False)
else:
    num_test_instances, test_data_loader = DataUtil.get_data_S_loader(which_data, batch_size=1, shuffle = False)


'''设置'''
path_checkpoint = "logs/14-torch-batch108-noLossWeight-RPNx2-ROIx2-MLP512-192S1ALL/Epoch155-Total_Loss0.1702-Val_Loss0.3879.pth"  # 断点路径
nms_iou = 0.01
score_thresh = 0.0
PLOT = True    #结果可视化
SHOW = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

Cuda = torch.cuda.is_available()

#加载自定义的模型
BACKBONE = "alexnet"
if dataset_name == "TEMPORAL":
    NUM_CLASSES = 6
else:
    NUM_CLASSES = 12

feat_stride = utils_base.get_feat_stride(IMAGE_SHAPE[1], BACKBONE)

backbone = AlexNet().features
backbone.out_channels = 384
anchor_generator = AnchorGenerator(sizes=((4*16,5*16,6*16,7*16,8*16,9*16,10*16),),
                                    aspect_ratios=((1.0),))

roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=(16,1),
                                                    sampling_ratio=0)
model = FasterRCNN(backbone,
                    num_classes=13,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler).to(device)

checkpoint = torch.load(path_checkpoint,  map_location=device)  # 加载断点
model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

model = model.eval()
np.set_printoptions(suppress=True)
ious_all = 0
detection_all = 0.0
acc = 0.0
cnt = 0
i = 0
my_map = {}
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
        
        # if labelV[0]==3.0 or labelV[0]==4.0:
        #     continue
        i+=1

        if False:
            mkey = int(int((bbox[0][1]-bbox[0][0])/10)*10)
            if not my_map.get(mkey, False):
                my_map.update({mkey:0})
            my_map[mkey] += 1
        torch.cuda.synchronize()   #增加同步操作
        start = time.time()

        predictions = model(dataV.unsqueeze(3))

        torch.cuda.synchronize() #增加同步操作
        end = time.time()
        _time += (end-start)

        if (predictions[1][0]['boxes'].shape[0]) == 0:
            continue
        if len(predictions[1])>1:
            print(len(predictions[1]))
        bbox = predictions[1][0]["boxes"][:,1:4:2].view(-1,2)
        conf = predictions[1][0]["scores"]
        label = predictions[1][0]["labels"]
        idx = 0

        max_iou = bbox_iou(np.asarray(bbox[idx].view(1,2).cpu()), np.asarray(bboxV.tolist()))[0][0]
        dete_acc = detection_acc(np.asarray(bbox[idx].view(1,2).cpu()), np.asarray(bboxV.tolist()))[0][0]
        
        if PLOT:
            X=np.linspace(0,192,192,endpoint=True)
            plt.contourf(dataV[0].cpu())
            plt.colorbar()

            currentAxis=plt.gca()
            if dataset_name=="TEMPORAL":
                h = 49
            else:
                h = 87
            rect=patches.Rectangle((bboxV[0][0], 1), bboxV[0][1]-bboxV[0][0] ,h, linewidth=2,edgecolor='white',facecolor='none')
            currentAxis.add_patch(rect)
            rect=patches.Rectangle((bbox[idx][0], 2), bbox[idx][1]-bbox[idx][0] ,h-2, linewidth=2,edgecolor='r',facecolor='none')
            currentAxis.add_patch(rect)
            
            plt.xlabel(which_data+str(i)+"  prd = "+actions[int(label[idx])]+"  gt = "+actions[int(labelV[0][0])]+"  acc = "+str(conf[idx])[:5]+"  iou = "+str(max_iou)[:5]+" wd = "+str(int(bboxV[0][1]-bboxV[0][0])))
            if max_iou >= 0.8 and max_iou<0.9:
                mkey = int(int((bboxV[0][1]-bboxV[0][0])/10)*10)
                if not my_map.get(mkey, False):
                    my_map.update({mkey:0})
                my_map[mkey] += 1
                plt.savefig("predict/%d.png"%(i), dpi=520)
                plt.show()
            plt.close()

        if SHOW:
            fe = model.extractor
            x = dataV.data
            fe.eval()
            for name, layer in fe._modules.items():
                x = layer(x)

                #if f'{name}'!='0' and f'{name}'!='3' and f'{name}'!= '5':
                 #   continue

                X=np.linspace(0,192,192,endpoint=True)
                plt.contourf(x[0].permute(1,0).cpu())
                plt.colorbar()
                currentAxis=plt.gca()
                plt.xlabel(which_data+str(i)+"  prd = "+actions[int(label[idx])]+"  gt = "+actions[int(labelV[0][0])]+"  acc = "+str(conf[idx])[:5]+"  iou = "+str(max_iou)[:5]+" wd = "+str(int(bboxV[0][1]-bboxV[0][0])))
                plt.savefig("feature_map_show/%d_layer_%s.png"%(i, f'{name}'), dpi=520)
                plt.show()
                plt.close()
            #break


        resultFile = open("input/detection-results/result"+str(i)+".txt", "w")

        j = idx
        resultFile.write("%s %.6f %f %f %f %f\n"%(actions[int(label[j])], conf[j], bbox[j][0], 0, bbox[j][1], 90))
        resultFile.close()
        groundFile = open("input/ground-truth/result"+str(i)+".txt", "w")
        for j in range(0, bboxV.shape[0]):
            groundFile.write("%s %f %f %f %f\n"%(actions[int(labelV[j][0])], bboxV[j][0], 0, bboxV[j][1], 90))
        groundFile.close()

        print("预测:", bbox[idx].tolist()," ", str(conf[idx].tolist())[:5], " ",label[idx], " 真实:",bboxV.tolist(), labelV.tolist(), "iou=", max_iou)
        if int(label[idx])==int(labelV[0]):
            acc = acc + 1
        ious_all += max_iou
        detection_all += dete_acc
        cnt+=1

ious_all /= i
detection_all /= i
acc /= i
_time /= i
print("有效预测：",cnt)
print("IOU: ",ious_all)
print("检测精度: ",detection_all)
print("分类准确度：", acc)
print("平均每份预测时间：", _time,"秒")

os.system('python get_map.py')
