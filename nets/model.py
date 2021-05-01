from nets.vgg16 import Vgg2FLow, decom_vgg16
from lib.Faster_RCNN import FasterRCNN
from torch.utils.data.dataset import Dataset
from nets.alexnet import AlexNet
from nets.unet_model import UNet
from lib.pool import MultiScaleRoIAlign
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator

def get_model(dataset, BACKBONE):
    if BACKBONE == "alexnet":
        #backbone = Feature(IMAGE_SHAPE[0], 384)    #双流
        #backbone,_ = decom_vgg16(dataset.image_shape[0])

        backbone = Vgg2FLow(dataset.image_shape[0])

        #backbone = AlexNet(n_channels=dataset.image_shape[0], n_classes=dataset.num_classes+1).features
    elif BACKBONE == "unet":
        backbone = UNet(n_channels=dataset.image_shape[0], n_classes=dataset.num_classes+1).features
    
    anchor_generator = AnchorGenerator(sizes=dataset.anchor,
                                    aspect_ratios=((1.0),))

    roi_pooler = MultiScaleRoIAlign(featmap_names=['0','1'],
                                                    output_size=(16,1),
                                                    sampling_ratio=0)
    model = FasterRCNN(backbone,
                    num_classes=dataset.num_classes+1,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)
    
    return model
