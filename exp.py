from numpy import dtype
from nets.alexnet import AlexNet
import torch
import torchvision
from lib.pool import MultiScaleRoIAlign
from lib.Faster_RCNN import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
# load a pre-trained model for classification and return
# only the features
backbone = AlexNet().features#torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
backbone.out_channels = 384

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((4,5,6,7,8,9,10),),
                                   aspect_ratios=((1.0),))

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be ['0']. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=(16,1),
                                                sampling_ratio=0)

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=12,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

x = torch.rand(1, 90, 192, 1)#[torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

# model.eval()
# predictions = model(x)
# print(predictions[0]['boxes'])

model.train()
targets = [
    {"boxes": torch.Tensor([[0,0,1,192],[0,0,1,192]]).view(2,4), 
     "labels":torch.Tensor([[1],[1]]).view(2,1).long()
    },
    {"boxes": torch.Tensor([[0,0,1,192]]).view(1,4),
    "labels":torch.Tensor([[1]]).view(1).long()
    }
    ]

losses = model(x, targets)
total_loss = losses['loss_rpn_box_reg']+ losses['loss_objectness'] + losses['loss_box_reg'] + losses['loss_classifier']
total_loss.backward()
print(losses)