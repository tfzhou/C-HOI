import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init

from mmdet.core import auto_fp16, force_fp32
from ..registry import HEADS
from ..utils import ConvModule, build_norm_layer
from mmdet.ops import DeformConv

import numpy as np
import pycocotools.mask as mask_util

@HEADS.register_module
class FusedSemanticHead_DCN(nn.Module):
    """Multi-level fused semantic segmentation head.

    in_1 -> deformable conv ---
                        |
    in_2 -> deformable conv -- |
                       ||
    in_3 -> deformable conv - ||
                      |||                  /-> 1x1 conv (mask prediction)
    in_4 -> deformable conv -----> 3x3 convs (*4)
                        |                  \-> 1x1 conv (feature)
    in_5 -> deformable conv ---
    """  # noqa: W605

    def __init__(self,
                 num_ins,
                 fusion_level,
                 num_convs=4,
                 in_channels=256,
                 conv_out_channels=256,
                 num_classes=183,
                 ignore_label=255,
                 loss_weight=0.2,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(FusedSemanticHead_DCN, self).__init__()
        self.num_ins = num_ins
        self.fusion_level = fusion_level
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        offset_channels = 18
        deformable_groups = 1
        dilation = 1
        self.dc1_offset = nn.Conv2d(
            self.in_channels,
            deformable_groups * offset_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation)
        self.dc1 = DeformConv(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            deformable_groups=deformable_groups,
            bias=False)
        self.dc2_offset = nn.Conv2d(
            self.in_channels,
            deformable_groups * offset_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation)
        self.dc2 = DeformConv(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            deformable_groups=deformable_groups,
            bias=False)
        self.norm2_name, self.norm2 = build_norm_layer(norm_cfg, self.in_channels, postfix=2)
        self.relu = nn.ReLU(inplace=False)

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else conv_out_channels
            self.convs.append(
                ConvModule(
                    in_channels,
                    conv_out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.conv_embedding = ConvModule(
            conv_out_channels,
            conv_out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        self.conv_logits = nn.Conv2d(conv_out_channels, self.num_classes, 1)

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)

    def init_weights(self):
        kaiming_init(self.conv_logits)

    @auto_fp16()
    def forward(self, feats):
        offset = self.dc1_offset(feats[self.fusion_level])
        x = self.dc1(feats[self.fusion_level], offset)
        offset = self.dc2_offset(x)
        x = self.dc2(x, offset)
        x = self.norm2(x)
        x = self.relu(x)

        fused_size = tuple(x.shape[-2:])
        for i, feat in enumerate(feats):
            if i != self.fusion_level:
                feat = F.interpolate(
                    feat, size=fused_size, mode='bilinear', align_corners=True)
                offset = self.dc1_offset(feat)
                feat = self.dc1(feat, offset)
                offset = self.dc2_offset(feat)
                feat = self.dc2(feat, offset)
                feat = self.norm2(feat)
                feat = self.relu(feat)
                x += feat

        for i in range(self.num_convs):
            x = self.convs[i](x)

        mask_pred = self.conv_logits(x)
        x = self.conv_embedding(x)
        return mask_pred, x

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, labels):
        labels = labels.squeeze(1).long()
        loss_semantic_seg = self.criterion(mask_pred, labels)
        loss_semantic_seg *= self.loss_weight
        return loss_semantic_seg

    def get_semantic_segm(self, semantic_pred):
        # only surport 1 batch
        segm_pred_map = F.softmax(semantic_pred, 1)
        segm_pred_map = torch.max(segm_pred_map, 1).indices
        segm_pred_map = segm_pred_map.float()
        segm_pred_map = segm_pred_map[0]

        segm_pred_map = segm_pred_map.cpu().numpy()

        return segm_pred_map