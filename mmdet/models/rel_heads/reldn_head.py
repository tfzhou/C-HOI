import logging

import torch
from torch import nn
import torch.nn.functional as F
from ..registry import HEADS
from .sparse_targets_rel import FrequencyBias
from ..builder import build_loss
from mmdet.core import force_fp32
from ..losses import accuracy, recall_topk, ReldnBCELoss

logger = logging.getLogger(__name__)


@HEADS.register_module
class ReldnHead(nn.Module):
    def __init__(self,
                 dim_in,
                 num_prd_classes,
                 use_freq_bias=False,
                 use_spatial_feat=True,
                 add_so_scores=True,
                 add_scores_all=True,
                 must_overlap=False,
                 mode='pic',
                 loss_cls=dict(
                     type='ReldnBCELoss',
                     use_sigmoid=True,
                     loss_weight=1.0)):
        super().__init__()
        dim_in_final = dim_in // 3
        self.dim_in_final = dim_in_final
        self.use_freq_bias = use_freq_bias
        self.use_spatial_feat = use_spatial_feat
        self.add_so_scores = add_so_scores
        self.add_scores_all = add_scores_all

        self.freq_bias = FrequencyBias(must_overlap=must_overlap, mode=mode)
            
        self.prd_cls_feats = nn.Sequential(
            nn.Linear(dim_in, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.1))
        self.prd_cls_scores = nn.Sequential(
            nn.Linear(1024, num_prd_classes),
            nn.Sigmoid())
        
        if use_spatial_feat:
            self.spt_cls_feats = nn.Sequential(
                nn.Linear(28, 64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, 64),
                nn.LeakyReLU(0.1))
            self.spt_cls_scores = nn.Sequential(
                nn.Linear(64, num_prd_classes),
                nn.Sigmoid())
        
        if add_so_scores:
            self.prd_sbj_scores = nn.Sequential(
                nn.Linear(dim_in_final, num_prd_classes),
                # nn.Linear(dim_in_final, 1024),
                # nn.LeakyReLU(0.1),
                # nn.Linear(1024, num_prd_classes),
                nn.Sigmoid())
            self.prd_obj_scores = nn.Sequential(
                nn.Linear(dim_in_final, num_prd_classes),
                # nn.Linear(dim_in_final, 1024),
                # nn.LeakyReLU(0.1),
                # nn.Linear(1024, num_prd_classes),
                nn.Sigmoid())

        self.loss_cls = build_loss(loss_cls)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # spo_feat will be concatenation of SPO
    def forward(self, spo_feat, spt_feat=None, sbj_labels=None,
                obj_labels=None, sbj_feat=None, obj_feat=None,
                run_baseline=False):

        if run_baseline:
            assert sbj_labels is not None and obj_labels is not None
            prd_bias_scores = self.freq_bias.rel_index_with_labels(
                torch.cat([sbj_labels-1, obj_labels-1], dim=1))
            prd_bias_scores = F.sigmoid(prd_bias_scores)

            return None, prd_bias_scores, None

        if spo_feat.dim() == 4:
            spo_feat = spo_feat.squeeze(3).squeeze(2)
        prd_cls_feats = self.prd_cls_feats(spo_feat)
        prd_vis_scores = self.prd_cls_scores(prd_cls_feats)

        if self.use_freq_bias:
            assert sbj_labels is not None and obj_labels is not None
            prd_bias_scores = self.freq_bias.rel_index_with_labels(
                torch.cat([sbj_labels-1, obj_labels-1], dim=1))
            prd_bias_scores = F.softmax(prd_bias_scores, dim=1)
        else:
            prd_bias_scores = None

        if self.use_spatial_feat:
            assert spt_feat is not None
            spt_cls_feats = self.spt_cls_feats(spt_feat)
            prd_spt_scores = self.spt_cls_scores(spt_cls_feats)
        else:
            prd_spt_scores = None
            
        if self.add_so_scores:
            assert sbj_feat is not None and obj_feat is not None
            prd_sbj_scores = self.prd_sbj_scores(sbj_feat)
            prd_obj_scores = self.prd_obj_scores(obj_feat)

            prd_vis_scores = (prd_vis_scores + prd_sbj_scores + prd_obj_scores) / 3

        return prd_vis_scores, prd_bias_scores, prd_spt_scores

    @force_fp32(apply_to=('rel_score',))
    def loss(self,
             cls_score,
             labels,
             label_weights,
             prd_spt_score=None,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            losses['loss_cls'] = self.loss_cls(cls_score, labels)
            if prd_spt_score is not None:
                losses['loss_spt'] = self.loss_cls(prd_spt_score, labels)

            losses['acc'] = recall_topk(cls_score, labels)
        return losses


