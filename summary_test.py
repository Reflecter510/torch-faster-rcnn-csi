import os
from dataset import S1P1, S2, TEMPORAL
import torch
from torch.autograd import Variable
import numpy as np
from utils.utils import detection_acc, bbox_iou, draw_bar, draw_table, locCls2Label, plot_confusion_matrix
from utils import DataUtil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
np.set_printoptions(4)
# 清除动作实例图       
os.system("rm predict/*.png")

# 使用 测试集test 或 训练集train
which_data = "test"

dataset_names = ["S1", "S2", "TEMPORAL"]
datasets = [S1P1, S2, TEMPORAL]

exps_index = [2*3-1, 6*3-1, 10*3-1, 12*3-1]
exps = ['exp0-alex', 'exp0-vgg', 
        'exp1-vgg-S0', 'exp1-vgg-S6',  'exp0-vgg', 'exp1-vgg-S24', 
        'exp2-vgg-ssn0', 'exp2-vgg-ssn1_12_1', 'exp0-vgg', 'exp2-vgg-ssn13_1210_13',
        'exp0-vgg', 'exp0-vgg']
methods = [
    ["AlexNet", "VGG"],
    ["不扩展", "s/6", "s/12", "s/24"],
    ["无", "(1)(1,2)(1)", "(1,3)(1,2,5)(1,3)", "(1,3)(1,2,10)(1,3)"],
    ["原始", "噪声增强"]
]

# 获取pkl文件列表
pkl_files = []
for exp in exps:
    for i in range(0, len(dataset_names)):
        dir = "结果/"+dataset_names[i]+"/"+exp
        files = os.listdir(dir)
        for each in files:
            if each[-3:] == "pkl":
                pkl_files.append([dir+"/"+each,i])

# 结果可视化
PLOT = False 
# 是否为每一个结果生成混淆矩阵  
gen_matrix = False

#------------------------------  模型  ----------------------------------------#
# 是否支持CUDA
Cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if Cuda else "cpu")
print("use" , device)


all_results = []
exp_result = []
raw_1 = []
raw_2 = []
_index = 0
for each in pkl_files:
    dataset = datasets[each[1]]
    # 根据数据集初始化相关变量
    dataset_name = dataset.name
    test_bacth = dataset.test_batch
    IMAGE_SHAPE = dataset.image_shape
    actions = dataset.actions

    if _index == exps_index[-1]-2:
        #print("test test noise!")
        DataUtil.noise = True
    #加载测试集
    num_test_instances, test_data_loader = DataUtil.get_data_loader(dataset_name, which_data, batch_size=test_bacth, shuffle = False)

    # 加载模型
    model = torch.load(each[0])
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

    pred_labels = []
    gt_labels = []

    for (data, bbox, label) in test_data_loader:
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
            predictions = model(dataV.unsqueeze(3))
            end = time.time()

            _time += (end-start)

            # 对批量预测的结果进行统计
            for idp in range(0, len(predictions[1])):
                prediction = predictions[1][idp]
                i+=1
                if prediction['boxes'].shape[0] == 0:
                    continue
                
                bbox = prediction["boxes"][:,1:4:2].view(-1,2)
                conf = prediction["scores"]
                label = prediction["labels"]
                idx = 0

                # 计算 IoU、逐帧检测精度、逐帧分类精度
                max_iou = bbox_iou(np.asarray(bbox[idx].view(1,2).cpu()), np.asarray(bboxV[idp].view(1,2)))[0][0]
                dete_acc = detection_acc(np.asarray(bbox[idx].view(1,2).cpu()), np.asarray(bboxV[idp].view(1,2)))[0][0]
                final_acc = detection_acc(np.asarray(bbox[idx].view(1,2).cpu()), np.asarray(bboxV[idp].view(1,2)), ACC=int(label[idx])==int(labelV[idp][0]))[0][0]

                if gen_matrix or (_index>2 and _index<6):
                    # 动作框转换为逐帧预测的标签
                    pred_labels.extend(locCls2Label(bboxV[idp].view(-1,2), labelV[idp][0].view(-1,1))[0])
                    gt_labels.extend(locCls2Label(bbox[idx].view(-1,2).cpu(), label[idx].view(-1,1))[0])

                # 绘制动作实例图
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
                    plt.savefig("predict/%d.png"%(i), dpi=98)
                    plt.close()
              
                # 统计结果
                if int(label[idx])==int(labelV[idp][0]):
                    acc = acc + 1
                ious_all += max_iou
                detection_all += dete_acc
                final_all += final_acc

                cnt+=1

    if gen_matrix or(_index>2 and _index<6):
        # 绘制分类混淆矩阵
        from sklearn.metrics import confusion_matrix
        matrix = confusion_matrix(gt_labels, pred_labels)
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        plot_confusion_matrix(matrix, dataset.actions,'confusion_matrix.png', title='Sample-level Action Classification Confusion Matrix')

        # 绘制检测混淆矩阵
        gt_labels = [1.0 if each>=1.0 else 0.0 for each in gt_labels]
        pred_labels = [1.0 if each>=1.0 else 0.0 for each in pred_labels]
        matrix = confusion_matrix(gt_labels, pred_labels)
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        plot_confusion_matrix(matrix, ["non","yes"],'detection_confusion_matrix.png', title='Sample-level Action Detection Confusion Matrix')

    ious_all /= num_test_instances
    detection_all /= num_test_instances
    final_all /= num_test_instances
    acc /= num_test_instances
    _time /= num_test_instances

    # 打印pkl文件名
    print(each[0])
    # print("有效预测：",cnt)
    # print("分类准确度：", str(acc)[:6], "  IOU: ",str(ious_all)[:6])
    # print("检测分类精度: ",str(final_all)[:6], " 检测精度: ",str(detection_all)[:6])
    # print("平均每份预测时间：", str(_time)[:6],"秒")

    # 保存所有结果到一个行向量
    raw_1.append(final_all)
    raw_1.append(detection_all)
    raw_2.append(acc)
    raw_2.append(ious_all)

    # 每三个数据集的结果作为一行
    if (_index+1) % 3 == 0:
        exp_result.append(raw_1 + raw_2)
        raw_1 = []
        raw_2 = []

    # 绘图
    for k in range(0, len(exps_index)):
        if _index==exps_index[k]:
            # 绘制分类准确度和IoU的柱状图
            accuracy = torch.Tensor(exp_result)[:,6::2]#.T.flatten()
            iou= torch.Tensor(exp_result)[:,7::2]#.T.flatten()
            draw_bar(["S1","S2","TEMPORAL"],accuracy.numpy(), methods[k], "分类准确度")
            draw_bar(["S1","S2","TEMPORAL"],iou.numpy(), methods[k], "IoU")

            # 绘制逐帧指标的表格，结果保留两位小数
            draw_table((torch.Tensor(exp_result)[:,:6]*10000).int().numpy()/100.0, methods[k])

            all_results.append(exp_result)
            exp_result = []

    _index += 1