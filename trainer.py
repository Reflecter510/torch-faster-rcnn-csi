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

    def forward(self, imgs, bboxes, labels, scale):
        bboxes = bboxes[0].view(-1, 2)
        
        targets = []
        for i in range(0, len(labels)):
            #TODO  将输入数据行列互换
            tmp = [ bboxes[i][0].tolist(), 0, bboxes[i][1].tolist(), 1]
            targets.append({"boxes": torch.Tensor(tmp).view(1,4), "labels": labels[i].long()})

        _losses = self.faster_rcnn(imgs.unsqueeze(2), targets)

        rpn_loc_loss = _losses['loss_rpn_box_reg']
        rpn_cls_loss = _losses['loss_objectness']
        roi_loc_loss = _losses['loss_box_reg']
        roi_cls_loss = _losses['loss_classifier']
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [104.03*rpn_loc_loss + 10*rpn_cls_loss + 0.94*roi_loc_loss + 4*roi_cls_loss]
        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        #losses.total_loss += self.reg_loss(self.faster_rcnn)
        losses.total_loss.backward()
        self.optimizer.step()
        return losses

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2   # 1^2
    in_weight = in_weight


    diff = in_weight * (x-t)
    abs_diff = diff.abs()    #|x-y|

    flag = (abs_diff.data < (1. / sigma2)).float()  #|x-y|<1
    y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape)
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1

    if pred_loc.is_cuda:
        pred_loc = pred_loc.cuda()
        gt_loc = gt_loc.cuda()
        in_weight = in_weight.cuda()
        
    # smooth_l1损失函数
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # 进行标准化
    loc_loss /= ((gt_label >= 0).sum().float())
    return loc_loss
