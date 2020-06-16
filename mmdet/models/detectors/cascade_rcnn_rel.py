from __future__ import division

import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, build_assigner, build_sampler,
                        merge_aug_masks, sample_pairs, bbox_union,
                        get_spatial_feature, bbox_mapping, merge_aug_bboxes,
                        multiclass_nms, assign_pairs)
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import RPNTestMixin
import torch.nn.functional as F
from mmdet.models.plugins import NonLocal2D
import cv2

import numpy as np


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

        self.theta = nn.Conv2d(256, 256, kernel_size=1)
        self.phi = nn.Conv2d(256, 256, kernel_size=1)

        self.ksi = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.theta(x)
        x2 = self.phi(x)

        n, c, h, w = x1.shape

        x1 = x1.view(n, c, -1)
        x1 = x1.permute(0, 2, 1)
        x2 = x2.view(n, c, -1)
        f = torch.matmul(x1, x2)
        f_div_C = F.softmax(f, dim=-1)

        xx = self.ksi(x).view(n, c, -1)
        xx = xx.permute(0, 2, 1)
        y = torch.matmul(f_div_C, xx)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(n, c, *x.size()[2:])

        y_down = F.upsample_bilinear(y, None, 0.5)

        z = y_down + x

        return z


@DETECTORS.register_module
class CascadeRCNNRel(BaseDetector, RPNTestMixin):

    def __init__(self,
                 num_stages,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 rel_head=None,
                 pretrained=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        super(CascadeRCNNRel, self).__init__()

        self.num_stages = num_stages
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = nn.ModuleList()
            self.bbox_head = nn.ModuleList()
            if not isinstance(bbox_roi_extractor, list):
                bbox_roi_extractor = [
                    bbox_roi_extractor for _ in range(num_stages)
                ]
            if not isinstance(bbox_head, list):
                bbox_head = [bbox_head for _ in range(num_stages)]
            assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
            for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
                self.bbox_roi_extractor.append(
                    builder.build_roi_extractor(roi_extractor))
                self.bbox_head.append(builder.build_head(head))

        if mask_head is not None:
            self.mask_head = nn.ModuleList()
            if not isinstance(mask_head, list):
                mask_head = [mask_head for _ in range(num_stages)]
            assert len(mask_head) == self.num_stages
            for head in mask_head:
                self.mask_head.append(builder.build_head(head))
            if mask_roi_extractor is not None:
                self.share_roi_extractor = False
                self.mask_roi_extractor = nn.ModuleList()
                if not isinstance(mask_roi_extractor, list):
                    mask_roi_extractor = [
                        mask_roi_extractor for _ in range(num_stages)
                    ]
                assert len(mask_roi_extractor) == self.num_stages
                for roi_extractor in mask_roi_extractor:
                    self.mask_roi_extractor.append(
                        builder.build_roi_extractor(roi_extractor))
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor

        self.reldn_head = builder.build_head(rel_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.sa = NonLocal2D(in_channels=256)

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(CascadeRCNNRel, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
            if self.with_mask:
                if not self.share_roi_extractor:
                    self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _rel_forward_test(self, x, det_bboxes, det_labels,
                          scale_factor, ori_shape,
                          filename=None,
                          im_width=None, im_height=None):

        assert det_labels.shape[0] == det_bboxes.shape[0]

        rel_test_cfg = self.test_cfg.rel
        run_baseline = rel_test_cfg.run_baseline

        # build pairs
        sbj_bboxes, sbj_labels, sbj_idxs, obj_bboxes, obj_labels, obj_idxs = \
            sample_pairs(det_bboxes, det_labels + 1,
                         overlap=True, overlap_th=0.4, test=True)

        ret = {'predictions': [], 'hoi_prediction': []}
        if sbj_bboxes is None or sbj_bboxes.shape[0] == 0:
            return ret

        num_pairs = len(sbj_idxs)

        # extract roi features
        with torch.no_grad():
            bbox_roi_extractor = self.bbox_roi_extractor[-1]
            bbox_rois = bbox2roi([det_bboxes])
            bbox_feats = bbox_roi_extractor(
                x[:len(bbox_roi_extractor.featmap_strides)], bbox_rois)

        sbj_feats = bbox_feats[sbj_idxs, ...]
        obj_feats = bbox_feats[obj_idxs, ...]
        union_bboxes = bbox_union(sbj_bboxes, obj_bboxes)
        union_rois = bbox2roi([union_bboxes])
        with torch.no_grad():
            union_feats = bbox_roi_extractor(
                x[:len(bbox_roi_extractor.featmap_strides)], union_rois)

        sbj_feats = sbj_feats.view(num_pairs, -1)
        obj_feats = obj_feats.view(num_pairs, -1)
        union_feats = union_feats.view(num_pairs, -1)

        assert sbj_feats.shape == obj_feats.shape == union_feats.shape
        visual_features = torch.cat([sbj_feats, obj_feats, union_feats],
                                    dim=-1)
        spatial_features = get_spatial_feature(sbj_bboxes, obj_bboxes,
                                               im_width, im_height)
        prd_vis_scores, prd_bias_scores, prd_spt_scores \
            = self.reldn_head(visual_features,
                              sbj_labels=sbj_labels,
                              obj_labels=obj_labels,
                              spt_feat=spatial_features,
                              sbj_feat=sbj_feats,
                              obj_feat=obj_feats,
                              run_baseline=run_baseline)

        # detect relations
        ret_bboxes = det_bboxes.cpu().numpy()
        ret_labels = det_labels.cpu().numpy()

        thresh = rel_test_cfg.thresh

        sbj_idxs = sbj_idxs.cpu().numpy()
        obj_idxs = obj_idxs.cpu().numpy()

        entity_index_to_output_index = {}
        unique_sbj_idxs = list(np.unique(sbj_idxs))
        unique_obj_idxs = list(np.unique(obj_idxs))
        unique_entity_idxs = unique_sbj_idxs + unique_obj_idxs
        for i, entity_index in enumerate(unique_entity_idxs):
            entity_index_to_output_index[entity_index] = i

            bbox = ret_bboxes[entity_index, :4]
            cat = ret_labels[entity_index]
            score = ret_bboxes[entity_index, -1]

            ret['predictions'].append({'bbox': bbox.tolist(),
                                       'category_id': str(cat + 1),
                                       'score': float(score)})

        if run_baseline:
            prd_bias_scores = prd_bias_scores.cpu().numpy()
            for i in range(prd_bias_scores.shape[0]):
                sbj_id = sbj_idxs[i]
                obj_id = obj_idxs[i]
                rel_scores = prd_bias_scores[i, :]
                rel_scores[0] = 0

                obj_cat = ret_labels[obj_id] + 1
                if obj_cat == 8 or obj_cat == 7 or obj_cat == 6:  # horse
                    rel_ids = [6]  # ride
                elif obj_cat == 9:
                    rel_ids = [8]
                else:
                    rel_ids = np.where(rel_scores > thresh)[0]

                sbj_output_index = entity_index_to_output_index[sbj_id]
                obj_output_index = entity_index_to_output_index[obj_id]

                for rel_id in rel_ids:
                    ret['hoi_prediction'].append({
                        'subject_id': int(sbj_output_index),
                        'object_id': int(obj_output_index),
                        'category_id': int(rel_id),
                        'score': float(rel_scores[rel_id])
                    })
        else:
            prd_vis_scores = prd_vis_scores.cpu().numpy()
            prd_bias_scores = prd_bias_scores.cpu().numpy()
            prd_bias_scores[:, 0] = 0
            prd_spt_scores = prd_spt_scores.cpu().numpy()

            prd_score = (prd_vis_scores + prd_spt_scores) * prd_bias_scores
            for i in range(prd_score.shape[0]):
                sbj_id = sbj_idxs[i]
                obj_id = obj_idxs[i]
                rel_scores = prd_score[i, :]

                sbj_score = float(det_bboxes[sbj_id, -1])
                obj_score = float(det_bboxes[obj_id, -1])

                if sbj_score < 0.4 or obj_score < 0.4:
                    continue

                obj_cat = ret_labels[obj_id] + 1
                rel_ids = np.where(rel_scores > thresh)[0]

                sbj_output_index = entity_index_to_output_index[sbj_id]
                obj_output_index = entity_index_to_output_index[obj_id]

                for rel_id in rel_ids:
                    ret['hoi_prediction'].append({
                        'subject_id': int(sbj_output_index),
                        'object_id': int(obj_output_index),
                        'category_id': int(rel_id),
                        'score': float(rel_scores[rel_id])
                    })

        return ret

    def _rel_forward_train(self,
                           x,
                           gt_bboxes,
                           gt_labels,
                           gt_rel,
                           gt_instid,
                           rel_train_cfg,
                           im_width=None,
                           im_height=None,
                           debug_image=None,
                           debug_filename=None):
        """
        :param x:
        :param gt_bboxes: n x 4 (tensor)
        :param gt_labels: n x 1 (tensor)
        :param det_bboxes: m x 5 (tensor)
        :param gt_rel: k x 3 (tensor)
        :param gt_instid: k x 1 (tensor)
        :param rel_train_cfg:
        :param filename: debug
        :return:
        """

        assert gt_bboxes.shape[0] == gt_labels.shape[0] == gt_instid.shape[0]

        combined_bboxes = gt_bboxes
        combined_labels = gt_labels

        sbj_bboxes, sbj_labels, sbj_idxs, obj_bboxes, obj_labels, obj_idxs = \
            sample_pairs(combined_bboxes, combined_labels)

        # extract mask features
        with torch.no_grad():
            bbox_roi_extractor = self.bbox_roi_extractor[-1]
            bbox_rois = bbox2roi([gt_bboxes])
            bbox_feats = bbox_roi_extractor(
                x[:len(bbox_roi_extractor.featmap_strides)], bbox_rois)

        # assign candidate pairs to a relation class
        # (including no-relationship class)
        relations, relation_targets = \
            assign_pairs(sbj_bboxes, sbj_labels, sbj_idxs,
                         obj_bboxes, obj_labels, obj_idxs,
                         gt_bboxes, gt_labels, gt_rel, gt_instid)

        num_relations = relations.shape[0]
        num_relation_cats = 11  # hard coded
        targets = relations.new_zeros((num_relations, num_relation_cats))
        for i in range(num_relations):
            targets[i, relation_targets[i]] = 1

        # get union bboxes
        sbj_bboxes = combined_bboxes[relations[:, -2].long(), :]
        obj_bboxes = combined_bboxes[relations[:, -1].long(), :]
        union_bboxes = bbox_union(sbj_bboxes, obj_bboxes)
        union_rois = bbox2roi([union_bboxes])

        # get visual features
        sbj_feats = bbox_feats[relations[:, -2].long(), ...]
        sbj_feats = self.sa(sbj_feats)

        obj_feats = bbox_feats[relations[:, -1].long(), ...]

        with torch.no_grad():
            union_feats = bbox_roi_extractor(
                x[:len(bbox_roi_extractor.featmap_strides)], union_rois)

        sbj_feats = sbj_feats.view(num_relations, -1)
        obj_feats = obj_feats.view(num_relations, -1)
        union_feats = union_feats.view(num_relations, -1)

        assert sbj_feats.shape == obj_feats.shape == union_feats.shape

        visual_features = torch.cat([sbj_feats, obj_feats, union_feats],
                                    dim=-1)

        spatial_features = get_spatial_feature(sbj_bboxes, obj_bboxes,
                                               im_width, im_height)

        prd_vis_scores, prd_bias_scores, prd_spt_scores \
            = self.reldn_head(visual_features,
                              sbj_labels=sbj_labels,
                              obj_labels=obj_labels,
                              spt_feat=spatial_features,
                              sbj_feat=sbj_feats,
                              obj_feat=obj_feats)

        label_weights = prd_vis_scores.new_ones(len(prd_vis_scores),
                                                dtype=torch.long)

        loss = self.reldn_head.loss(prd_vis_scores, targets, label_weights,
                                    prd_spt_score=prd_spt_scores)

        return loss

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_instid=None,
                      gt_rel=None):
        x = self.extract_feat(img)

        losses = dict()

        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
                bbox_sampler = build_sampler(
                    rcnn_train_cfg.sampler, context=self)
                num_imgs = img.size(0)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                            rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = bbox_head(bbox_feats)

            bbox_targets = bbox_head.get_target(sampling_results, gt_bboxes,
                                                gt_labels, rcnn_train_cfg)
            loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                if not self.share_roi_extractor:
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    pos_rois = bbox2roi(
                        [res.pos_bboxes for res in sampling_results])
                    mask_feats = mask_roi_extractor(
                        x[:mask_roi_extractor.num_inputs], pos_rois)
                    if self.with_shared_head:
                        mask_feats = self.shared_head(mask_feats)
                else:
                    # reuse positive bbox feats
                    pos_inds = []
                    device = bbox_feats.device
                    for res in sampling_results:
                        pos_inds.append(
                            torch.ones(
                                res.pos_bboxes.shape[0],
                                device=device,
                                dtype=torch.uint8))
                        pos_inds.append(
                            torch.zeros(
                                res.neg_bboxes.shape[0],
                                device=device,
                                dtype=torch.uint8))
                    pos_inds = torch.cat(pos_inds)
                    mask_feats = bbox_feats[pos_inds]
                mask_head = self.mask_head[i]
                mask_pred = mask_head(mask_feats)
                mask_targets = mask_head.get_target(sampling_results, gt_masks,
                                                    rcnn_train_cfg)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                roi_labels = bbox_targets[0]  # bbox_targets is a tuple
                with torch.no_grad():
                    proposal_list = bbox_head.refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)

        batch_size = len(img)
        rel_losses = None
        for i in range(batch_size):
            im_height, im_width, _ = img_meta[i]['img_shape']
            # now only support batch with size 1
            rel_loss = self._rel_forward_train(x,
                                               gt_bboxes=gt_bboxes[i],
                                               gt_labels=gt_labels[i],
                                               gt_rel=gt_rel[i],
                                               gt_instid=gt_instid[i],
                                               rel_train_cfg=self.train_cfg.rel,
                                               im_width=im_width,
                                               im_height=im_height)
            if rel_losses is None:
                rel_losses = rel_loss
            else:
                for name, value in rel_loss.items():
                    rel_losses[name] += value
        for name, value in rel_losses.items():
            losses['rel.{}'.format(name)] = (value / batch_size)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        img_shape = img_meta[0]['img_shape']
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            bbox_feats = bbox_roi_extractor(
                x[:len(bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            cls_score, bbox_pred = bbox_head(bbox_feats)
            ms_scores.append(cls_score)

            if self.test_cfg.keep_all_stages:
                det_bboxes, det_labels = bbox_head.get_det_bboxes(
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
                bbox_result = bbox2result(det_bboxes, det_labels,
                                          bbox_head.num_classes)
                ms_bbox_result['stage{}'.format(i)] = bbox_result

                if self.with_mask:
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    mask_head = self.mask_head[i]
                    if det_bboxes.shape[0] == 0:
                        mask_classes = mask_head.num_classes - 1
                        segm_result = [[] for _ in range(mask_classes)]
                    else:
                        _bboxes = (
                            det_bboxes[:, :4] *
                            scale_factor if rescale else det_bboxes)
                        mask_rois = bbox2roi([_bboxes])
                        mask_feats = mask_roi_extractor(
                            x[:len(mask_roi_extractor.featmap_strides)],
                            mask_rois)
                        if self.with_shared_head:
                            mask_feats = self.shared_head(mask_feats, i)
                        mask_pred = mask_head(mask_feats)
                        segm_result = mask_head.get_seg_masks(
                            mask_pred, _bboxes, det_labels, rcnn_test_cfg,
                            ori_shape, scale_factor, rescale)
                    ms_segm_result['stage{}'.format(i)] = segm_result

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_meta[0])

        cls_score = sum(ms_scores) / self.num_stages
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                mask_classes = self.mask_head[-1].num_classes - 1
                segm_result = [[] for _ in range(mask_classes)]
            else:
                _bboxes = (
                    det_bboxes[:, :4] *
                    scale_factor if rescale else det_bboxes)
                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                for i in range(self.num_stages):
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    mask_feats = mask_roi_extractor(
                        x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                    if self.with_shared_head:
                        mask_feats = self.shared_head(mask_feats)
                    mask_pred = self.mask_head[i](mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks,
                                               [img_meta] * self.num_stages,
                                               self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result

        if self.with_rel:
            im_height, im_width, _ = img_meta[0]['img_shape']
            filename = img_meta[0]['filename']
            relation_preds = self._rel_forward_test(x,
                                                    det_bboxes,
                                                    det_labels,
                                                    scale_factor,
                                                    ori_shape,
                                                    filename=filename,
                                                    im_width=im_width,
                                                    im_height=im_height)
            relation_preds['file_name'] = filename

        if not self.test_cfg.keep_all_stages:
            if self.with_mask:
                results = (ms_bbox_result['ensemble'],
                           ms_segm_result['ensemble'])
            elif self.with_rel:
                results = (ms_bbox_result['ensemble'],
                           relation_preds)
            else:
                results = ms_bbox_result['ensemble']
        else:
            if self.with_mask:
                results = {
                    stage: (ms_bbox_result[stage], ms_segm_result[stage])
                    for stage in ms_bbox_result
                }
            else:
                results = ms_bbox_result

        return results

    def aug_test(self, imgs, img_metas, proposals=None, rescale=False):
        """Test with augmentations.
                If rescale is False, then returned bboxes and masks will fit the scale
                of imgs[0].
                """

        ms_bbox_result = {}
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)

        rcnn_test_cfg = self.test_cfg.rcnn
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(self.extract_feats(imgs), img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_roi_extractor = self.bbox_roi_extractor[i]
                bbox_head = self.bbox_head[i]
                bbox_feats = bbox_roi_extractor(
                    x[:len(bbox_roi_extractor.featmap_strides)], rois)
                if self.with_shared_head:
                    bbox_feats = self.shared_head(bbox_feats)
                cls_score, bbox_pred = bbox_head(bbox_feats)
                ms_scores.append(cls_score)

                if i < self.num_stages - 1:
                    bbox_label = cls_score.argmax(dim=1)
                    rois = bbox_head.regress_by_class(rois, bbox_label,
                                                      bbox_pred, img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_rel:
            ori_shape = img_meta[0]['ori_shape']
            im_height, im_width, _ = img_meta[0]['img_shape']
            filename = img_meta[0]['filename']
            relation_preds = self._rel_forward_test(x,
                                                    det_bboxes,
                                                    det_labels,
                                                    scale_factor,
                                                    ori_shape,
                                                    im_width=im_width,
                                                    im_height=im_height)
            relation_preds['file_name'] = filename

        return (ms_bbox_result, relation_preds)

    def show_result(self, data, result, img_norm_cfg, **kwargs):
        if self.with_mask:
            if len(result) == 2:
                ms_bbox_result, ms_segm_result = result
                if isinstance(ms_bbox_result, dict):
                    result = (ms_bbox_result['ensemble'],
                              ms_segm_result['ensemble'])
            else:
                ms_bbox_result, ms_segm_result, ms_sema_result = result
                if isinstance(ms_bbox_result, dict):
                    result = (ms_bbox_result['ensemble'],
                              ms_segm_result['ensemble'],
                              ms_sema_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        super(CascadeRCNNRel, self).show_result(data, result, img_norm_cfg,
                                                **kwargs)
