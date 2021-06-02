import os
from dataset import S1P1, S2, TEMPORAL
import torch
from torch.autograd import Variable
import numpy as np
from utils.utils import detection_acc, bbox_iou, draw_bar, draw_table, label2LocCls, locCls2Label, plot_confusion_matrix, slide_window
from utils import DataUtil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
np.set_printoptions(4)
# 清除动作实例图       
os.system("rm predict/*.png")

# 使用 测试集test 或 训练集train
which_data = "test"

# 实验和数据集配置
dataset_names = ["S1", "S2", "TEMPORAL"]
datasets = [S1P1, S2, TEMPORAL]
# 区分每一种实验的下标
exps_index = [3*3-1, 5*3-1, 9*3-1, 13*3-1, 15*3-1]
exps = ['SlideWindow', 'Unet', 'exp0-vgg',
        'exp0-alex', 'exp0-vgg', 
        'exp1-vgg-S0', 'exp1-vgg-S6',  'exp0-vgg', 'exp1-vgg-S24', 
        'exp2-vgg-ssn0', 'exp2-vgg-ssn1_12_1', 'exp0-vgg', 'exp2-vgg-ssn13_1210_13',
        'exp0-vgg', 'exp0-vgg']
methods = [
    ["滑动窗口", "Unet", "本文模型"],
    ["AlexNet", "VGG"],
    ["不扩展", "s/6", "s/12", "s/24"],
    ["无", "(1)(1,2)(1)", "(1,3)(1,2,5)(1,3)", "(1,3)(1,2,10)(1,3)"],
    ["原始", "噪声增强"]
]

# 获取pkl模型文件列表
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
device = torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")
print("use" , device)

# 单个实验结果矩阵
exp_result = []
# 逐帧分类精度和逐帧检测精度的单行结果向量
raw_1 = []
# 动作分类准确度和IoU的单行结果向量
raw_2 = []

# pkl模型文件列表的下标
_index = 0
for each in pkl_files:
    # 根据数据集初始化相关变量
    dataset = datasets[each[1]]
    dataset_name = dataset.name
    test_bacth = dataset.test_batch
    IMAGE_SHAPE = dataset.image_shape
    actions = dataset.actions

    # 对最后三份数据集进行噪声增强测试
    if _index == exps_index[-1]-2:
        DataUtil.noise = True

    # 加载测试集
    num_test_instances, test_data_loader = DataUtil.get_data_loader(dataset_name, which_data, batch_size=test_bacth, shuffle = False)

    # 加载模型
    if each[0][-8:] == "null.pkl":
        # 滑动窗口方法不需要加载模型，直接调用函数slide_window
        model = slide_window
    else:
        model = torch.load(each[0], map_location=device)
        # 测试模式
        model = model.eval()

    # 初始化单个模型的准确度
    np.set_printoptions(suppress=True)
    ious_all = 0
    detection_all = 0.0
    final_all = 0.0
    acc = 0.0
    # 逐帧的预测结果
    unet_result = []
    pred_labels = []
    gt_labels = []
    i = 0

    # 取出一批数据
    for (data, bbox, label) in test_data_loader:
        with torch.no_grad():
            data = data.reshape([-1,IMAGE_SHAPE[0], IMAGE_SHAPE[1]])
            dataV = Variable(data.to(device))
            bboxV = Variable(bbox.to(device))
            labelV = Variable(label.to(device))

            # 批量预测
            if each[0][-8:] == "null.pkl":
                # 滑动窗口
                predictions = model(dataV.permute(0,2,1))
            elif each[0][-8:] == "unet.pkl":
                # Unet
                unet_result = model(dataV).data.max(1)[1]
                predictions = label2LocCls(unet_result)
            else:
                # 本文模型
                predictions = model(dataV.unsqueeze(3))

            # 对批量预测的结果进行统计
            for idp in range(0, len(predictions[1])):
                prediction = predictions[1][idp]
                i+=1
                # 跳过空结果
                if prediction['boxes'].shape[0] == 0:
                    continue
                
                # 预测框
                bbox = prediction["boxes"][:,1:4:2].view(-1,2)
                conf = prediction["scores"]
                label = prediction["labels"]
                idx = 0
                pred_tensor = bbox[idx].view(1,2)
                # 真实框
                gt_tensor = bboxV[idp].view(1,2)
                # 动作类别是否预测正确
                ACC = int(label[idx])==int(labelV[idp][0])

                # 计算 IoU、逐帧检测精度、逐帧分类精度
                max_iou = bbox_iou(pred_tensor.numpy(), gt_tensor.numpy())[0][0]
                dete_acc = detection_acc(pred_tensor.numpy(), gt_tensor.numpy())[0][0]
                final_acc = detection_acc(pred_tensor.numpy(), gt_tensor.numpy(), ACC=ACC)[0][0]

                if gen_matrix or (_index>11 and _index<15):
                    # 动作框转换为逐帧预测的标签
                    pred_labels.extend(locCls2Label(pred_tensor, label[idx].view(-1,1))[0])
                gt_labels.extend(locCls2Label(gt_tensor, labelV[idp][0].view(-1,1))[0])

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
                    plt.savefig("predict/%d.png"%(i), dpi=98)
                    plt.close()
              
                # 统计结果
                if ACC:
                    acc = acc + 1
                ious_all += max_iou
                detection_all += dete_acc
                final_all += final_acc

    if gen_matrix or(_index>11 and _index<15):
        # 绘制分类混淆矩阵
        from sklearn.metrics import confusion_matrix
        matrix = confusion_matrix(gt_labels, pred_labels, normalize='true')
        plot_confusion_matrix(matrix, dataset.actions,'confusion_matrix.png', title='Sample-level Action Classification Confusion Matrix')

        # 绘制检测混淆矩阵
        gt_labels = [1.0 if each>=1.0 else 0.0 for each in gt_labels]
        pred_labels = [1.0 if each>=1.0 else 0.0 for each in pred_labels]
        matrix = confusion_matrix(gt_labels, pred_labels, normalize='true')
        plot_confusion_matrix(matrix, ["non","yes"],'detection_confusion_matrix.png', title='Sample-level Action Detection Confusion Matrix')

    # 计算平均结果
    ious_all /= num_test_instances
    detection_all /= num_test_instances
    final_all /= num_test_instances
    acc /= num_test_instances

    # unet的逐帧分类精度直接计算，因为unet结果转换为动作框后的逐帧类别有误差
    if unet_result != []:
        final_all = unet_result.eq(torch.Tensor(gt_labels).view(-1,192)).sum()/ (num_test_instances * 192)
    # 滑动窗口方法的结果没有逐帧分类精度
    elif each[0][-8:] == "null.pkl":
        final_all = 0
    
    # 打印pkl文件名
    print(each[0])
    # print("分类准确度：", str(acc)[:6], "  IOU: ",str(ious_all)[:6])
    # print("检测分类精度: ",str(final_all)[:6], " 检测精度: ",str(detection_all)[:6])

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

            exp_result = []

    _index += 1