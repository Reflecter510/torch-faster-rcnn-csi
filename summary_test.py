import os
from dataset import S1P1, S2, TEMPORAL
import torch
from torch.autograd import Variable
import numpy as np
from utils.utils import detection_acc, bbox_iou, draw_bar, draw_table, label2LocCls, locCls2Label, plot_confusion_matrix, plot_csi_box_actions, slide_window
from utils import DataUtil
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
PLOT = True 
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
    # 根据指定数据集初始化相关变量
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
        # 加载pkl文件
        model = torch.load(each[0], map_location=device)
        # 测试模式
        model = model.eval()

    # 初始化单个模型的准确度
    np.set_printoptions(suppress=True)
    # IoU
    ious_all = 0
    # 逐帧检测精度
    detection_all = 0.0
    # 逐帧分类精度
    class_acc_all = 0.0
    # 动作分类准确度
    accuracy = 0.0

    # 逐帧的预测结果
    unet_result = []
    pred_labels = []
    gt_labels = []

    # CSI序列数
    i = 0

    # 取出一批数据
    for (data, bbox, label) in test_data_loader:
        with torch.no_grad():
            data = data.reshape([-1,IMAGE_SHAPE[0], IMAGE_SHAPE[1]])
            # CSI序列
            dataV = Variable(data.to(device))
            # 真实动作框
            bboxV = Variable(bbox.to(device))
            # 真实动作标签
            labelV = Variable(label.to(device))

            # 批量预测
            if each[0][-8:] == "null.pkl":
                # 滑动窗口
                predictions = model(dataV.permute(0,2,1))
            elif each[0][-8:] == "unet.pkl":
                # Unet
                unet_result.append(model(dataV).data.max(1)[1])
                # 逐帧预测结果转换为动作框
                predictions = label2LocCls(unet_result[-1])
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
                bbox, conf, label = prediction["boxes"][:,1:4:2].view(-1,2)[0], prediction["scores"][0], prediction["labels"][0]
                pred_box = bbox.view(1,2)
                pred_action = int(label)
                # 真实框
                gt_box = bboxV[idp].view(1,2)
                gt_action = int(labelV[idp][0])
                # 动作类别是否预测正确
                ACC = pred_action==gt_action

                # 计算 IoU、逐帧检测精度、逐帧分类精度
                max_iou = bbox_iou(pred_box.numpy(), gt_box.numpy())[0][0]
                dete_acc = detection_acc(pred_box.numpy(), gt_box.numpy())[0][0]
                class_acc = detection_acc(pred_box.numpy(), gt_box.numpy(), ACC=ACC)[0][0]

                if gen_matrix or (_index>11 and _index<15):
                    # 动作框转换为逐帧预测的标签，仅在生成混淆矩阵时使用
                    pred_labels.extend(locCls2Label(pred_box, label.view(-1,1))[0])
                # 真实动作框转换为逐帧的标签
                gt_labels.extend(locCls2Label(gt_box, labelV[idp][0].view(-1,1))[0])

                # 绘制动作实例图
                if PLOT and _index == 14 and (i==270 or i==140 or i==130 or i==119 or i==69):
                    plot_csi_box_actions(dataV[idp].cpu(), 
                                        pred_box[0], gt_box[0], actions[pred_action], actions[gt_action], 
                                        str(conf.tolist())[:5], dataset_name, i)
              
                # 统计结果
                if ACC:
                    accuracy = accuracy + 1
                ious_all += max_iou
                detection_all += dete_acc
                class_acc_all += class_acc

    # 混淆矩阵
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
    class_acc_all /= num_test_instances
    accuracy /= num_test_instances

    # unet的逐帧分类精度直接计算，因为unet结果转换为动作框后的逐帧类别有误差
    if unet_result != []:
        unet_result = torch.cat(unet_result, dim=0)
        class_acc_all = unet_result.eq(torch.Tensor(gt_labels).view(-1,192)).sum()/ (num_test_instances * 192)
    # 滑动窗口方法的结果没有逐帧分类精度
    elif each[0][-8:] == "null.pkl":
        class_acc_all = 0
    
    # 打印pkl文件名
    print(each[0])

    # 保存所有结果到一个行向量
    raw_1.append(class_acc_all)
    raw_1.append(detection_all)
    raw_2.append(accuracy)
    raw_2.append(ious_all)

    # 每三个数据集的结果作为一行
    if (_index+1) % 3 == 0:
        exp_result.append(raw_1 + raw_2)
        raw_1 = []
        raw_2 = []

    # 绘制每个实验的柱状图和表格
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