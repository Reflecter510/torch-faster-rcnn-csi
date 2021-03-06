import torch
from torch import nn
from torch.nn.init import xavier_uniform
import math
import numpy as np
# 实现时间金字塔池化操作

class Identity(torch.nn.Module):
    def forward(self, input):
        return input


def parse_stage_config(stage_cfg):
    if isinstance(stage_cfg, int):
        return (stage_cfg,), stage_cfg
    elif isinstance(stage_cfg, tuple) or isinstance(stage_cfg, list):
        return stage_cfg, sum(stage_cfg)
    else:
        raise ValueError("Incorrect STPP config {}".format(stage_cfg))

# 时间金字塔池化
class StructuredTemporalPyramidPooling(torch.nn.Module):
    """
    This the STPP operator for training. Please see the ICCV paper for more details.
    """
    def __init__(self, standalong_classifier=False, configs=(1, (1,2), 1)):
        super(StructuredTemporalPyramidPooling, self).__init__()
        self.sc = standalong_classifier

        # 结构配置，总长度
        starting_parts, starting_mult = parse_stage_config(configs[0])
        course_parts, course_mult = parse_stage_config(configs[1])
        ending_parts, ending_mult = parse_stage_config(configs[2])

        self.feat_multiplier = starting_mult + course_mult + ending_mult
        self.parts = (starting_parts, course_parts, ending_parts)
        self.norm_num = (starting_mult, course_mult, ending_mult)

    # tensor(n, 14, :), st和ed缩放因子tensor(n,2), int -> [st, st+cr, st+cr+ed
    # ft: 特征向量
    # scaling: 缩放因子（不使用）
    # seg_split: 阶段划分  [持续阶段起点, 结束阶段起点, 总长度]
    def forward(self, ft, scaling, seg_split):
        x1 = seg_split[0]
        x2 = seg_split[1]
        n_seg = seg_split[2]
        ft_dim = ft.size()[1]

        src = ft.view(-1, n_seg, ft_dim)
        #scaling = scaling.view(-1, 2)
        n_sample = src.size()[0]

        # 计算单个阶段的时间金字塔池化结果
        # stage_ft: 该阶段的特征向量
        # stage_parts: 单个阶段的所有池化尺寸，例如(1,3)
        # norm_num: 阶段总长度
        def get_stage_stpp(stage_ft, stage_parts, norm_num, scaling):
            stage_stpp = []
            stage_len = stage_ft.size(1)
            # 对于单个阶段的所有池化尺寸，例如(1,3)
            for n_part in stage_parts:
                ticks = torch.arange(0, stage_len + 1e-5, stage_len / n_part)
                # 为每个尺寸生成池化后的特征向量
                for i in range(n_part):
                    # 时间金字塔关键操作
                    part_ft = stage_ft[:, int(ticks[i]):int(ticks[i+1]), :].mean(dim=1) / norm_num
                    if scaling is not None:
                        part_ft = part_ft * scaling.resize(n_sample, 1)
                    stage_stpp.append(part_ft)
            return stage_stpp

        feature_parts = []
        feature_parts.extend(get_stage_stpp(src[:, :x1, :], self.parts[0], self.norm_num[0], None))  # starting
        feature_parts.extend(get_stage_stpp(src[:, x1:x2, :], self.parts[1], self.norm_num[1], None))  # course
        feature_parts.extend(get_stage_stpp(src[:, x2:, :], self.parts[2], self.norm_num[2], None))  # ending
        stpp_ft = torch.cat(feature_parts, dim=1)
        if not self.sc:
            return stpp_ft, stpp_ft
        else:
            course_ft = src[:, x1:x2, :].mean(dim=1)
            return course_ft, stpp_ft

######################## 以下代码均没有使用
class STPPReorgainzed:
    """
        This class implements the reorganized testing in SSN.
        It can accelerate the testing process by transforming the matrix multiplications into simple pooling.
    """

    def __init__(self, feat_dim,
                 act_score_len, comp_score_len, reg_score_len,
                 standalong_classifier=False, with_regression=True, stpp_cfg=(1, 1, 1)):
        self.sc = standalong_classifier
        self.act_len = act_score_len
        self.comp_len = comp_score_len
        self.reg_len = reg_score_len
        self.with_regression = with_regression
        self.feat_dim = feat_dim

        starting_parts, starting_mult = parse_stage_config(stpp_cfg[0])
        course_parts, course_mult = parse_stage_config(stpp_cfg[1])
        ending_parts, ending_mult = parse_stage_config(stpp_cfg[2])

        feature_multiplie = starting_mult + course_mult + ending_mult
        self.stpp_cfg = (starting_parts, course_parts, ending_parts)

        self.act_slice = slice(0, self.act_len if self.sc else (self.act_len * feature_multiplie))
        self.comp_slice = slice(self.act_slice.stop, self.act_slice.stop + self.comp_len * feature_multiplie)
        self.reg_slice = slice(self.comp_slice.stop, self.comp_slice.stop + self.reg_len * feature_multiplie)

    def forward(self, scores, proposal_ticks, scaling):
        assert scores.size(1) == self.feat_dim
        n_out = proposal_ticks.size(0)

        out_act_scores = torch.zeros((n_out, self.act_len)).cuda()
        raw_act_scores = scores[:, self.act_slice]

        out_comp_scores = torch.zeros((n_out, self.comp_len)).cuda()
        raw_comp_scores = scores[:, self.comp_slice]

        if self.with_regression:
            out_reg_scores = torch.zeros((n_out, self.reg_len)).cuda()
            raw_reg_scores = scores[:, self.reg_slice]
        else:
            out_reg_scores = None
            raw_reg_scores = None

        def pspool(out_scores, index, raw_scores, ticks, scaling, score_len, stpp_cfg):
            offset = 0
            for stage_idx, stage_cfg in enumerate(stpp_cfg):
                if stage_idx == 0:
                    s = scaling[0]
                elif stage_idx == len(stpp_cfg) - 1:
                    s = scaling[1]
                else:
                    s = 1.0

                stage_cnt = sum(stage_cfg)
                left = ticks[stage_idx]
                right = max(ticks[stage_idx] + 1, ticks[stage_idx + 1])

                if right <= 0 or left >= raw_scores.size(0):
                    offset += stage_cnt
                    continue
                for n_part in stage_cfg:
                    part_ticks = np.arange(left, right + 1e-5, (right - left) / n_part)
                    for i in range(n_part):
                        pl = int(part_ticks[i])
                        pr = int(part_ticks[i+1])
                        if pr - pl >= 1:
                            out_scores[index, :] += raw_scores[pl:pr,
                                                    offset * score_len: (offset + 1) * score_len].mean(dim=0) * s
                        offset += 1

        for i in range(n_out):
            ticks = proposal_ticks[i].numpy()
            if self.sc:
                try:
                    out_act_scores[i, :] = raw_act_scores[ticks[1]:max(ticks[1] + 1, ticks[2]), :].mean(dim=0)
                except:
                    print(ticks)
                    raise

            else:
                pspool(out_act_scores, i, raw_act_scores, ticks, scaling[i], self.act_len, self.stpp_cfg)

            pspool(out_comp_scores, i, raw_comp_scores, ticks, scaling[i], self.comp_len, self.stpp_cfg)

            if self.with_regression:
                pspool(out_reg_scores, i, raw_reg_scores, ticks, scaling[i], self.reg_len, self.stpp_cfg)

        return out_act_scores, out_comp_scores, out_reg_scores


class OHEMHingeLoss(torch.autograd.Function):
    """
    This class is the core implementation for the completeness loss in paper.
    It compute class-wise hinge loss and performs online hard negative mining (OHEM).
    """

    @staticmethod
    def forward(ctx, pred, labels, is_positive, ohem_ratio, group_size):
        n_sample = pred.size()[0]
        assert n_sample == len(labels), "mismatch between sample size and label size"
        losses = torch.zeros(n_sample)
        slopes = torch.zeros(n_sample)
        for i in range(n_sample):
            losses[i] = max(0, 1 - is_positive * pred[i, labels[i] - 1])
            slopes[i] = -is_positive if losses[i] != 0 else 0

        losses = losses.view(-1, group_size).contiguous()
        sorted_losses, indices = torch.sort(losses, dim=1, descending=True)
        keep_num = int(group_size * ohem_ratio)
        loss = torch.zeros(1).cuda()
        for i in range(losses.size(0)):
            loss += sorted_losses[i, :keep_num].sum()
        ctx.loss_ind = indices[:, :keep_num]
        ctx.labels = labels
        ctx.slopes = slopes
        ctx.shape = pred.size()
        ctx.group_size = group_size
        ctx.num_group = losses.size(0)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        labels = ctx.labels
        slopes = ctx.slopes

        grad_in = torch.zeros(ctx.shape)
        for group in range(ctx.num_group):
            for idx in ctx.loss_ind[group]:
                loc = idx + group * ctx.group_size
                grad_in[loc, labels[loc] - 1] = slopes[loc] * grad_output.data[0]
        return torch.autograd.Variable(grad_in.cuda()), None, None, None, None


class CompletenessLoss(torch.nn.Module):
    def __init__(self, ohem_ratio=0.17):
        super(CompletenessLoss, self).__init__()
        self.ohem_ratio = ohem_ratio

        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, labels, sample_split, sample_group_size):
        pred_dim = pred.size()[1]
        pred = pred.view(-1, sample_group_size, pred_dim)
        labels = labels.view(-1, sample_group_size)

        pos_group_size = sample_split
        neg_group_size = sample_group_size - sample_split
        pos_prob = pred[:, :sample_split, :].contiguous().view(-1, pred_dim)
        neg_prob = pred[:, sample_split:, :].contiguous().view(-1, pred_dim)
        pos_ls = OHEMHingeLoss.apply(pos_prob, labels[:, :sample_split].contiguous().view(-1), 1,
                                     1.0, pos_group_size)
        neg_ls = OHEMHingeLoss.apply(neg_prob, labels[:, sample_split:].contiguous().view(-1), -1,
                                     self.ohem_ratio, neg_group_size)
        pos_cnt = pos_prob.size(0)
        neg_cnt = int(neg_prob.size()[0] * self.ohem_ratio)

        return pos_ls / float(pos_cnt + neg_cnt) + neg_ls / float(pos_cnt + neg_cnt)


class ClassWiseRegressionLoss(torch.nn.Module):
    """
    This class implements the location regression loss for each class
    """

    def __init__(self):
        super(ClassWiseRegressionLoss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(self, pred, labels, targets):
        indexer = labels.data # - 1
        prep = pred #[:, indexer, :]
        class_pred = torch.cat((torch.diag(prep[:, :,  0]).view(-1, 1),
                                torch.diag(prep[:, :, 1]).view(-1, 1)),
                               dim=1)
        loss = self.smooth_l1_loss(class_pred.view(-1).cuda(), targets.contiguous().view(-1).cuda()) * 2
        return loss
