from __future__ import  absolute_import
#from utils.l1_norm import Regularization
#from nets.ssn_ops import ClassWiseRegressionLoss
import os
import time
from collections import namedtuple
from torch.nn import functional as F
from torch import nn
import torch as torch

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn,optimizer):
        super(FasterRCNNTrainer, self).__init__()
        self.faster_rcnn = faster_rcnn
        self.optimizer = optimizer

    def forward(self, imgs, bboxes, labels):
        bboxes = bboxes[0].view(-1, 2)
        
        targets = []
        for i in range(0, len(labels)):
            tmp = [0, bboxes[i][0].tolist(), 1, bboxes[i][1].tolist()]
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            targets.append({"boxes": torch.Tensor(tmp).view(1,4).to(device), "labels": labels[i].long()})

        _losses = self.faster_rcnn(imgs.unsqueeze(3), targets)

        rpn_loc_loss = _losses['loss_rpn_box_reg']
        rpn_cls_loss = _losses['loss_objectness']
        roi_loc_loss = _losses['loss_box_reg']
        roi_cls_loss = _losses['loss_classifier']
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss]
        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels)
        #losses.total_loss += self.reg_loss(self.faster_rcnn)
        losses.total_loss.backward()
        self.optimizer.step()
        return losses
