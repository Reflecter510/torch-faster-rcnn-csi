import torch
import torch.nn as nn
import torchvision
from torchvision.models.utils import load_state_dict_from_url
import torch.nn.functional as F

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
#--------------------------------------#
#   VGG16的结构
#--------------------------------------#
class VGG(nn.Module):
    def __init__(self, features, num_classes=13, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # 平均池化到7x7大小
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        #--------------------------------------#
        #   分类部分
        #--------------------------------------#
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 平均池化
        x = self.avgpool(x)
        # 平铺后
        x = torch.flatten(x, 1)
        # 分类部分
        x = self.classifier(x)
        # x = F.softmax(x, dim=-1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

#--------------------------------------#
#   特征提取部分
#--------------------------------------#
def make_layers(cfg, in_channels, batch_norm=True):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(3,1), padding=(1,0))
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def decom_vgg16(in_channels):
    model = VGG(make_layers(cfg,in_channels))
    
    # 获取特征提取部分
    features = list(model.features)[:43]
    # 获取分类部分
    classifier = model.classifier
    classifier = list(classifier)
    # 除去Dropout部分
    del classifier[6]
    del classifier[5]
    del classifier[2]
    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    
    features.out_channels = 512

    # print(classifier)
    return features,classifier
'''features,roi_head_classifier'''

''' define loss function for ROI loss and RPN loss '''
class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

    def forward(self, x, y):
        k = torch.nn.functional.cross_entropy(x, y.long(), ignore_index=-1)
        return k

if __name__ == '__main__':
    vgg,cls = decom_vgg16()
    print(vgg)
    data = torch.randn((1, 90, 192, 1))
    data = torch.autograd.Variable(data)
    out = vgg.forward(data)
    print(out.shape)