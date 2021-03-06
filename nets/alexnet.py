
from collections import OrderedDict
from nets.unet_model import UNet_features
from torch.functional import Tensor
import torch.nn as nn
import torch
from torch.nn.functional import selu
import torchvision.models.detection.faster_rcnn

# AlexNet的一个卷积模块
class CnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool = False):
        super().__init__()
        self.pool = pool
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size,1), stride=(stride,1), padding=(padding,0))
        self.BN = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3,1), stride=(2,1))

    def forward(self, x):
        x = self.layer(x)
        
        x = self.relu(x)
        if self.pool :
            x = self.maxpool(x)

        x = self.BN(x)
        return x 

# 基于AlexNet的三流结构
class Feature(nn.Module):
    def __init__(self, n_channels=52, out_channels=384):
        super().__init__()
        # 作为Faster RCNN的主干网络时 必须定义输出通道数
        self.out_channels = out_channels
        self.layer1 = CnnBlock(n_channels, self.out_channels, kernel_size=11, stride=4, padding=2, pool=True)
        self.layer2 = CnnBlock(n_channels, self.out_channels, kernel_size=5, stride=1, padding=2)
        self.layer3 = CnnBlock(n_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        #self.layer4 = UNet_features(n_channels=n_channels)#CnnBlock(n_channels, 384, kernel_size=11, stride=4, padding=2, pool=True)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)

        #x4 = self.layer4(x)
        
        return OrderedDict([('0', x1),('1', x2), ('2', x3)])

class AlexNet(nn.Module):
    out_channels = 384
    def __init__(self, n_channels=52, n_classes=7):
        super(AlexNet, self).__init__()
       
        # 特征提取，只有这个用于Faster RCNN
        self.features = nn.Sequential(
            CnnBlock(n_channels, 128, kernel_size=11, stride=4, padding=2, pool=True),
            CnnBlock(128, 192, kernel_size=5, stride=1, padding=2),
            CnnBlock(192, self.out_channels, kernel_size=3, stride=1, padding=1)
        )
        # 作为Faster RCNN的主干网络时 必须定义输出通道数
        self.features.out_channels = self.out_channels
        
        # 平均池化到7x7大小
        self.avgpool = nn.AdaptiveAvgPool1d(6)
        self.classifier = nn.Sequential(
            nn.Linear(384 * 1 * 16, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        # x = x.view(x.size(0), 256 * 6 * 6)
        # 平均池化
        x = self.avgpool(x)
        # 平铺后
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def decom_alexnet():
    model = AlexNet()

    # 获取特征提取部分
    features = list(model.features)

    # 获取分类部分
    classifier = model.classifier[:-1]
    classifier = list(classifier)

    features = nn.Sequential(*features)
    # print(features)
    classifier = nn.Sequential(*classifier)
    # print(classifier)
    return features,classifier


if __name__ == '__main__':
    # Example
    data = torch.rand((1,90,192,1))
    data = torch.autograd.Variable(data)

    net = AlexNet(n_channels=90)
    print(net)

    features = net.features
    #print(features)

    out = features(data)  #[1, 256, 1, 92]

    x = data
    model = features
    
    # 绘制各个卷积层输出的特征图
    from tensorboardX import SummaryWriter
    import torchvision.utils as vutils
    from torch.nn import functional as F
    f_writer = SummaryWriter('./feature_map_show')   # 数据存放在这个文件夹
    img_grid = vutils.make_grid(x, normalize=True, scale_each=True, nrow=2)
    # 绘制原始图像
    f_writer.add_image('raw img', img_grid, global_step=666)  # j 表示feature map数
    print(x.size())

    model.eval()
    for name, layer in model._modules.items():

        x = layer(x)
        print(f'{name}')
        if  'layer' in name or 'conv' in name:
            x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
            img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=4)  # normalize进行归一化处理
            f_writer.add_image(f'{name}_feature_maps', img_grid, global_step=0)
    f_writer.close()

    print("end")
    