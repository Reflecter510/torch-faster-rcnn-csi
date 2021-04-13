
from torch.functional import Tensor
import torch.nn as nn
import torch
import torchvision.models.detection.faster_rcnn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.out_channels = 384

        self.features = nn.Sequential(
            nn.Conv2d(90, 128, kernel_size=(11,1), stride=(4,1), padding=(2,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(2,1)),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 192, kernel_size=(5,1), padding=(2,0)), #尺寸不变
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 384, kernel_size=(3,1), padding=(1,0)), #尺寸不变
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
        )
        # 平均池化到7x7大小
        self.avgpool = nn.AdaptiveAvgPool1d(6)
        self.classifier = nn.Sequential(
            nn.Linear(384 * 1 * 16, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
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

# alexnet 根据数据尺寸获取步长
def get_alex_feat_stride(x):
    ori = x*1.0
    x = ((x+2*2-(11-1)-1)/4+1)
    
    # maxpool   ks=3  stride=2
    x = (x-(3-1)-1)/2+1
    
    x = ((x+2*2-(5-1)-1)+1)

    # maxpool   ks=3  stride=2
    #x = (x-(3-1)-1)/2+1
    
    return ori/int(x)

if __name__ == '__main__':
    # Example
    data = torch.rand((1,90,192,1))
    data = torch.autograd.Variable(data)
    print(get_alex_feat_stride(192))

    net = AlexNet()
    print(net)

    features,classifier = decom_alexnet()
    #print(features)

    out = features(data)  #[1, 256, 1, 92]

    x = data
    model = features
    
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
    