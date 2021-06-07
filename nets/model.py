from nets.vgg16 import Vgg2FLow, decom_vgg16
from lib.Faster_RCNN import FasterRCNN
from torch.utils.data.dataset import Dataset
from nets.alexnet import AlexNet
from nets.unet_model import UNet
from lib.pool import MultiScaleRoIAlign
from torch import nn
#from torchvision.models.detection.anchor_utils import AnchorGenerator
from lib.rpn import AnchorGenerator

# 封装Faster RCNN 模型
def get_model(dataset, BACKBONE):
    n_channels = dataset.image_shape[0]
    n_classes = dataset.num_classes + 1
    # 主干特征提取网络
    if BACKBONE == "vgg":
        backbone,_ = decom_vgg16(n_channels)
    elif BACKBONE == "alexnet":
        #backbone = Feature(n_channels, 384)    #扁平alex三流
        #backbone = Vgg2FLow(n_channels)        #vgg alex 双流
        backbone = AlexNet(n_channels, n_classes).features
    elif BACKBONE == "unet":
        backbone = UNet(n_channels, n_classes).features
    
    # 锚框生成器
    anchor_generator = AnchorGenerator(sizes=dataset.anchor,
                                    aspect_ratios=((1.0),))

    # 统一池化层  池化尺寸 16x1
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=(16,1),
                                                    sampling_ratio=0)
    model = FasterRCNN(backbone,
                    num_classes=n_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)
    
    return model
