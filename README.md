# 基于CSI的动作检测 Faster-RCNN模型
# 运行summary_test.py即可获得所有实验结果

# 工程目录说明
lib 存放自定义修改的Faster R-CNN代码（基于torch官方库）

nets 存放主干特征提取网络、时间金字塔

utils 存放数据集读取、动作框结果转换相关工具函数

logs 存放模型训练断点、训练损失的tensorboard日志

logs_map 存放模型测试结果的tensorboard日志

结果  存放论文实验结果（pkl文件）

predict 存放预测实例图片

unet 仅用于summary_test.py

venv 可运行的环境（windows）

> **已弃用的目录**
> feature_map_show 卷积层特征图输出
> input 存放预测结果和真实结果的txt版本（绘制mAP时使用）

# 数据集根目录
DataUtil.home_dir = '../frcnn_csi/'

# 训练并测试所有模型断点
运行train.py

# 单个模型测试
运行model_test.py