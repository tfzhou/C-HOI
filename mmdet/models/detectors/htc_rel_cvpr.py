import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (bbox2result, bbox2roi, build_assigner, build_sampler,
                        merge_aug_masks, sample_pairs, bbox_union,
                        bbox_mapping, multiclass_nms, merge_aug_bboxes,
                        assign_pairs, tensor2imgs, get_spatial_feature)
from mmdet.datasets import PicDatasetV20
from .. import builder
from ..registry import DETECTORS
from .cascade_rcnn import CascadeRCNN

import os
import numpy as np


@DETECTORS.register_module
class HybridTaskCascadeRelCVPR(CascadeRCNN):

    def __init__(self,
                 num_stages,
                 backbone,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 semantic_fusion=('bbox', 'mask'),
                 interleaved=True,
                 use_mask_feat=True,
                 use_feat_fusion=False,
                 fusion_method=None,
                 mask_info_flow=True,
                 rel_head=None,
                 rel_binary_head=None,
                 rel_neck=None,
                 **kwargs):
        super(HybridTaskCascadeRelCVPR, self).__init__(num_stages, backbone,
                                                   **kwargs)
        assert self.with_bbox and self.with_mask
        assert not self.with_shared_head  # shared head not supported
        if semantic_head is not None:
            self.semantic_roi_extractor = builder.build_roi_extractor(
                semantic_roi_extractor)
            self.semantic_head = builder.build_head(semantic_head)

        self.semantic_fusion = semantic_fusion
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow

        test_cfg = kwargs.get('test_cfg')
        if 'save_feat' in test_cfg:
            self.save_feat = test_cfg['save_feat']
            self.save_folder = test_cfg['save_folder']
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)
        else:
            self.save_feat = None
            self.save_folder = None

        if 'rel_save_folder' in test_cfg:
            self.rel_save_folder = test_cfg['rel_save_folder']
            if not os.path.exists(self.rel_save_folder):
                os.makedirs(self.rel_save_folder)
        else:
            self.rel_save_folder = None

        try:
            self.rel_cfg = self.train_cfg.get('rel')
            self.debug = self.rel_cfg.debug
        except:
            pass

        self.use_mask_feat = use_mask_feat
        self.use_feat_fusion = use_feat_fusion
        self.fusion_method = fusion_method

        # relation feature extraction
        #self.rel_backbone = builder.build_backbone(backbone)
        #self.with_rel_neck = rel_neck is not None
        #if rel_neck is not None:
        #    self.rel_neck = builder.build_neck(rel_neck)

        self.reldn_head = builder.build_head(rel_head)
        self.reldn_binary_head = builder.build_head(rel_binary_head)

    def extract_feat_rel(self, img):
        x = self.rel_backbone(img)
        if self.rel_neck:
            x = self.neck(x)
        return x

    @property
    def with_semantic(self):
        if hasattr(self, 'semantic_head') and self.semantic_head is not None:
            return True
        else:
            return False

    def _bbox_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            rcnn_train_cfg,
                            semantic_feat=None,
                            ret_score=False):

        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat

        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_targets = bbox_head.get_target(sampling_results, gt_bboxes,
                                            gt_labels, rcnn_train_cfg)
        loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
        if ret_score is False:
            return loss_bbox, rois, bbox_targets, bbox_pred
        else:
            return loss_bbox, rois, bbox_targets, bbox_pred, cls_score

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            semantic_feat=None):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        pos_rois)

        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             pos_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat

        # mask information flow
        # forward all previous mask heads to obtain last_feat, and fuse it
        # with the normal mask feature
        if self.mask_info_flow:
            last_feat = None
            for i in range(stage):
                last_feat = self.mask_head[i](
                    mask_feats, last_feat, return_logits=False)
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
        else:
            mask_pred = mask_head(mask_feats)

        mask_targets = mask_head.get_target(sampling_results, gt_masks,
                                            rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
        return loss_mask

    def _bbox_forward_test(self, stage, x, rois, semantic_feat=None,
                           return_feat=False):
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(
            x[:len(bbox_roi_extractor.featmap_strides)], rois)
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat
        cls_score, bbox_pred = bbox_head(bbox_feats)
        if return_feat is True:
            return cls_score, bbox_pred, bbox_feats
        else:
            return cls_score, bbox_pred

    def _mask_forward_test(self, stage, x, bboxes, semantic_feat=None):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_rois = bbox2roi([bboxes])
        mask_feats = mask_roi_extractor(
            x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             mask_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat
        if self.mask_info_flow:
            last_feat = None
            last_pred = None
            for i in range(stage):
                mask_pred, last_feat = self.mask_head[i](mask_feats, last_feat)
                if last_pred is not None:
                    mask_pred = mask_pred + last_pred
                last_pred = mask_pred
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
            if last_pred is not None:
                mask_pred = mask_pred + last_pred
        else:
            mask_pred = mask_head(mask_feats)
        return mask_pred

    def _rel_forward_test(self,
                          x,
                          det_bboxes,
                          det_labels,
                          det_masks,
                          scale_factor,
                          ori_shape,
                          im_width=None,
                          im_height=None,
                          semantic_feat=None,
                          use_mask_feat=True,
                          use_feat_fusion=False,
                          fusion_method=None):

        assert det_labels.shape[0] == det_bboxes.shape[0] == det_masks.shape[0]
        if isinstance(det_masks, torch.Tensor):
            det_masks = det_masks.sigmoid().cpu().numpy()

        rel_test_cfg = self.test_cfg.rel
        run_baseline = rel_test_cfg.run_baseline

        # build pairs
        sbj_bboxes, sbj_labels, sbj_idxs, obj_bboxes, obj_labels, obj_idxs =\
            sample_pairs(det_bboxes, det_labels + 1)

        if sbj_bboxes is None:
            return None

        thresh = rel_test_cfg.thresh

        num_pairs = len(sbj_idxs)

        # extract roi features
        with torch.no_grad():
            if use_feat_fusion is True:
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_rois = bbox2roi([det_bboxes])
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                        mask_semantic_feat = F.adaptive_avg_pool2d(
                            mask_semantic_feat, mask_feats.shape[-2:])
                    mask_feats += mask_semantic_feat

                bbox_roi_extractor = self.bbox_roi_extractor[-1]
                bbox_rois = bbox2roi([det_bboxes])
                bbox_feats = bbox_roi_extractor(
                    x[:len(bbox_roi_extractor.featmap_strides)], bbox_rois)
                if self.with_semantic and 'bbox' in self.semantic_fusion:
                    bbox_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], bbox_rois)
                    if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                        bbox_semantic_feat = F.adaptive_avg_pool2d(
                            bbox_semantic_feat, bbox_feats.shape[-2:])
                    bbox_feats += bbox_semantic_feat

                bbox_feats = F.upsample_bilinear(bbox_feats, None, 2.0)

                if fusion_method == 'max':
                    feats = torch.max(bbox_feats, mask_feats)
                else:
                    feats = bbox_feats + mask_feats

                # get visual features
                sbj_feats = feats[sbj_idxs, ...]
                obj_feats = feats[obj_idxs, ...]

            elif use_mask_feat is True:
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_rois = bbox2roi([det_bboxes])
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                        mask_semantic_feat = F.adaptive_avg_pool2d(
                            mask_semantic_feat, mask_feats.shape[-2:])
                    mask_feats += mask_semantic_feat

                # get visual features
                sbj_feats = mask_feats[sbj_idxs, ...]
                obj_feats = mask_feats[obj_idxs, ...]
            else:
                bbox_roi_extractor = self.bbox_roi_extractor[-1]
                bbox_rois = bbox2roi([det_bboxes])
                bbox_feats = bbox_roi_extractor(
                    x[:len(bbox_roi_extractor.featmap_strides)], bbox_rois)
                if self.with_semantic and 'bbox' in self.semantic_fusion:
                    bbox_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], bbox_rois)
                    if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                        bbox_semantic_feat = F.adaptive_avg_pool2d(
                            bbox_semantic_feat, bbox_feats.shape[-2:])
                    bbox_feats += bbox_semantic_feat

                # get visual features
                sbj_feats = bbox_feats[sbj_idxs, ...]
                obj_feats = bbox_feats[obj_idxs, ...]


        # union
        union_bboxes = bbox_union(sbj_bboxes, obj_bboxes)
        union_rois = bbox2roi([union_bboxes])
        with torch.no_grad():
            if use_feat_fusion is True:
                union_feats_mask = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], union_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    union_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], union_rois)
                    if union_semantic_feat.shape[-2:] != union_feats_mask.shape[-2:]:
                        union_semantic_feat = F.adaptive_avg_pool2d(
                            union_semantic_feat, union_feats_mask.shape[-2:])
                    union_feats_mask += union_semantic_feat

                union_feats = bbox_roi_extractor(
                    x[:len(bbox_roi_extractor.featmap_strides)], union_rois)
                if self.with_semantic and 'bbox' in self.semantic_fusion:
                    union_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], union_rois)
                    if union_semantic_feat.shape[-2:] != union_feats.shape[-2:]:
                        union_semantic_feat = F.adaptive_avg_pool2d(
                            union_semantic_feat, union_feats.shape[-2:])
                    union_feats += union_semantic_feat

                union_feats = F.upsample_bilinear(union_feats, None, 2.0)
                if fusion_method == 'max':
                    union_feats = torch.max(union_feats, union_feats_mask)
                else:
                    union_feats = union_feats + union_feats_mask
            elif use_mask_feat:
                union_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], union_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    union_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], union_rois)
                    if union_semantic_feat.shape[-2:] != union_feats.shape[-2:]:
                        union_semantic_feat = F.adaptive_avg_pool2d(
                            union_semantic_feat, union_feats.shape[-2:])
                    union_feats += union_semantic_feat
            else:
                union_feats = bbox_roi_extractor(
                    x[:len(bbox_roi_extractor.featmap_strides)], union_rois)
                if self.with_semantic and 'bbox' in self.semantic_fusion:
                    union_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], union_rois)
                    if union_semantic_feat.shape[-2:] != union_feats.shape[-2:]:
                        union_semantic_feat = F.adaptive_avg_pool2d(
                            union_semantic_feat, union_feats.shape[-2:])
                    union_feats += union_semantic_feat

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
        ret_masks = det_masks
        ret_relations = []

        # fill ret_relations
        if run_baseline:
            thresh = rel_test_cfg.thresh

            sbj_idxs = sbj_idxs.cpu().numpy()
            obj_idxs = obj_idxs.cpu().numpy()
            prd_bias_scores = prd_bias_scores.cpu().numpy()
            # never predict __no_relation__ for frequency prior
            for i in range(prd_bias_scores.shape[0]):
                sbj_id = sbj_idxs[i]
                obj_id = obj_idxs[i]
                rel_scores = prd_bias_scores[i, :]
                rel_scores[0] = 0
                rel_ids = np.where(rel_scores > thresh)

                ret_relations.extend(
                    [[sbj_id, obj_id, rel_id, rel_scores[rel_id]]
                     for rel_id in rel_ids])
        else:
            prd_vis_scores = prd_vis_scores.cpu().numpy()
            if prd_bias_scores is not None:
                prd_bias_scores = prd_bias_scores.cpu().numpy()
                prd_bias_scores[:, 0] = 0
            prd_spt_scores = prd_spt_scores.cpu().numpy()

            #prd_score = (prd_vis_scores + prd_spt_scores) * prd_bias_scores
            #prd_score = (prd_vis_scores + prd_spt_scores + prd_bias_scores) / 3
            prd_score = prd_vis_scores + prd_spt_scores

            for i in range(prd_score.shape[0]):
                sbj_id = sbj_idxs[i]
                obj_id = obj_idxs[i]
                rel_scores = prd_score[i, :]
                rel_scores[0] = 0
                #rel_ids = rel_scores.argsort()[-5:]

                sbj_score = float(det_bboxes[sbj_id, -1])
                obj_score = float(det_bboxes[obj_id, -1])

                # remove low-confidence objects
                if sbj_score < 0.4 or obj_score < 0.4:
                    continue

                sorted_indexs = np.argsort(rel_scores)[::-1]
                rel_ids = sorted_indexs[:3]

                #rel_ids = np.where(rel_scores > thresh)[0]

                ret_relations.extend(
                    [[sbj_id, obj_id, rel_ids, rel_scores[rel_ids],
                      sbj_bboxes[i, :].data.cpu().numpy(),
                      obj_bboxes[i, :].data.cpu().numpy()]])

        ret = {
            'ret_bbox': ret_bboxes,
            'ret_mask': ret_masks,
            'ret_label': ret_labels,
            'ret_relation': ret_relations,
            'scale_factor': scale_factor,
            'ori_shape': ori_shape
        }

        return ret

    def _rel_forward_train_ranking(self, x, gt_bboxes, gt_labels, gt_rel,
                                   gt_instid,
                                   im_width=None, im_height=None,
                                   debug_image=None, debug_filename=None,
                                   use_mask_feat=True):

        assert gt_bboxes.shape[0] == gt_labels.shape[0] == gt_instid.shape[0]

        sbj_bboxes, sbj_labels, sbj_idxs, obj_bboxes, obj_labels, obj_idxs = \
            sample_pairs(gt_bboxes, gt_labels)


        # extract mask features
        mask_roi_extractor = self.mask_roi_extractor[-1]
        bbox_roi_extractor = self.bbox_roi_extractor[-1]
        with torch.no_grad():
            if use_mask_feat:
                mask_rois = bbox2roi([gt_bboxes])
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
            else:
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
        num_relation_cats = 31  # hard coded
        targets = relations.new_zeros((num_relations, num_relation_cats))
        for i in range(num_relations):
            targets[i, relation_targets[i]] = 1

        sbj_bboxes = gt_bboxes[relations[:, -2].long(), :4]
        obj_bboxes = gt_bboxes[relations[:, -1].long(), :4]
        union_bboxes = bbox_union(sbj_bboxes, obj_bboxes)
        union_rois = bbox2roi([union_bboxes])

        with torch.no_grad():
            if use_mask_feat:
                union_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], union_rois)
            else:
                union_feats = bbox_roi_extractor(
                    x[:len(bbox_roi_extractor.featmap_strides)], union_rois)

        ## debug
        if debug_image is not None:
            import os
            import matplotlib.pyplot as plt
            debug_dir = os.path.join('./tmp/{}'.format(debug_filename[:-4]))
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)

            debug_num_relation = relations.shape[0]
            for i in range(debug_num_relation):
                sbj_label = relations[i, 0].to(dtype=torch.int)
                obj_label = relations[i, 1].to(dtype=torch.int)
                rel_label = relations[i, 2].to(dtype=torch.int)
                sbj_index = relations[i, 3].to(dtype=torch.int)
                obj_index = relations[i, 4].to(dtype=torch.int)

                sbj_bbox = gt_bboxes[sbj_index, :4]
                obj_bbox = gt_bboxes[obj_index, :4]

                union_bbox = union_bboxes[i, :4]

                # plot
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(debug_image, cmap=plt.cm.gray)

                x1, y1, x2, y2 = sbj_bbox
                ax.add_artist(plt.Rectangle((x1, y1), x2-x1+1, y2-y1+1,
                                            fill=False, color='r'))
                sbj_text = PicDatasetV20.CATEGORIES[sbj_label-1]['name']
                ax.add_artist(plt.Text(x1, y1, sbj_text, size='x-large',
                                       color='r'))

                x1, y1, x2, y2 = obj_bbox
                ax.add_artist(plt.Rectangle((x1, y1), x2-x1+1, y2-y1+1,
                                            fill=False, color='g'))
                obj_text = PicDatasetV20.CATEGORIES[obj_label-1]['name']
                ax.add_artist(plt.Text(x1, y1, obj_text, size='x-large',
                                       color='g'))

                x1, y1, x2, y2 = union_bbox
                ax.add_artist(plt.Rectangle((x1, y1), x2-x1+1, y2-y1+1,
                                            fill=False, color='w'))
                rel_text = PicDatasetV20.REL_CATEGORIES[rel_label-1]['name']\
                    if rel_label > 0 else 'None'
                ax.add_artist(plt.Text(x1, y1, rel_text, size='x-large',
                                       color='w',
                                       bbox=dict(facecolor='k', alpha=0.5)))


                #
                rel_text = PicDatasetV20.REL_CATEGORIES[rel_label-1]['name'] \
                    if rel_label > 0 else 'None'
                title = '<{}, {}, {}>'.format(sbj_text, obj_text, rel_text)
                ax.set_title(title)
                ax.axis('off')

                savename = os.path.join(debug_dir, '{:05d}.png'.format(i))
                plt.savefig(savename, dpi=100)
                plt.close()

        # get visual features
        if use_mask_feat is True:
            sbj_feats = mask_feats[relations[:, -2].long(), ...]
            obj_feats = mask_feats[relations[:, -1].long(), ...]
        else:
            sbj_feats = bbox_feats[relations[:, -2].long(), ...]
            obj_feats = bbox_feats[relations[:, -1].long(), ...]

        num_relations = relations.shape[0]
        sbj_feats = sbj_feats.view(num_relations, -1)
        obj_feats = obj_feats.view(num_relations, -1)
        union_feats = union_feats.view(num_relations, -1)

        vis_feats = torch.cat([sbj_feats, obj_feats, union_feats], dim=-1)
        spt_feats = get_spatial_feature(sbj_bboxes, obj_bboxes,
                                        im_width, im_height)

        scores, embedding = self.reldn_binary_head(vis_feats, spt_feats)

        # build ranking pairs
        x_pos = []
        x_neg = []
        targets_numpy = targets.cpu().data.numpy()
        for i in range(num_relations):
            if targets_numpy[i][0] == 0:
                x_neg.append(i)
            else:
                x_pos.append(i)

        if len(x_pos) == 0 or len(x_neg) == 0:
            loss = {'loss': torch.tensor(0).float().to(scores.get_device())}
        else:
            embedding_pos = embedding[x_pos, :]
            scores_pos = scores[x_pos]
            embedding_neg = embedding[x_neg, :]
            scores_neg = scores[x_neg]

            # sample scores
            scores_pos = scores_pos.repeat(len(x_neg), 1)
            scores_neg = scores_neg.repeat_interleave(len(x_pos), 0)

            max_sample = min(64, len(x_neg) * len(x_pos))
            sample_indexs = np.random.choice(len(x_neg)*len(x_pos), max_sample)
            scores_pos = scores_pos[sample_indexs]
            scores_neg = scores_neg[sample_indexs]

            # sample features
            embedding_pos = embedding_pos.repeat(len(x_neg), 1)
            embedding_neg = embedding_neg.repeat_interleave(len(x_pos), 0)
            embedding_pos = embedding_pos[sample_indexs, :]
            embedding_neg = embedding_neg[sample_indexs, :]

            # compute loss
            targets_ranking = embedding_neg.new_ones(max_sample, 1)
            targets_cosine = embedding_neg.new_ones(max_sample, 1) * -1
            loss = self.reldn_binary_head.loss(scores_pos, embedding_pos,
                                               scores_neg, embedding_neg,
                                               targets_ranking, targets_cosine)

        return loss

    def _rel_forward_train(self,
                           x,
                           gt_bboxes,
                           gt_labels,
                           gt_rel,
                           gt_instid,
                           rel_train_cfg,
                           semantic_feat=None,
                           im_width=None,
                           im_height=None,
                           debug_image=None,
                           debug_filename=None,
                           use_mask_feat=True,
                           use_feat_fusion=False,
                           fusion_method=None):
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
        sbj_bboxes, sbj_labels, sbj_idxs, obj_bboxes, obj_labels, obj_idxs =\
            sample_pairs(combined_bboxes, combined_labels)

        # extract mask features
        with torch.no_grad():
            if use_feat_fusion is True:
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_rois = bbox2roi([combined_bboxes])
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                        mask_semantic_feat = F.adaptive_avg_pool2d(
                            mask_semantic_feat, mask_feats.shape[-2:])
                    mask_feats += mask_semantic_feat

                bbox_roi_extractor = self.bbox_roi_extractor[-1]
                bbox_rois = bbox2roi([combined_bboxes])
                bbox_feats = bbox_roi_extractor(
                    x[:len(bbox_roi_extractor.featmap_strides)], bbox_rois)
                if self.with_semantic and 'bbox' in self.semantic_fusion:
                    bbox_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], bbox_rois)
                    if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                        bbox_semantic_feat = F.adaptive_avg_pool2d(
                            bbox_semantic_feat, bbox_feats.shape[-2:])
                    bbox_feats += bbox_semantic_feat

                bbox_feats = F.upsample_bilinear(bbox_feats, None, 2.0)

                if fusion_method == 'max':
                    feats = torch.max(bbox_feats, mask_feats)
                else:
                    feats = bbox_feats + mask_feats

            elif use_mask_feat is True:
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_rois = bbox2roi([combined_bboxes])
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                        mask_semantic_feat = F.adaptive_avg_pool2d(
                            mask_semantic_feat, mask_feats.shape[-2:])
                    mask_feats += mask_semantic_feat
            else:
                bbox_roi_extractor = self.bbox_roi_extractor[-1]
                bbox_rois = bbox2roi([combined_bboxes])
                bbox_feats = bbox_roi_extractor(
                    x[:len(bbox_roi_extractor.featmap_strides)], bbox_rois)
                if self.with_semantic and 'bbox' in self.semantic_fusion:
                    bbox_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], bbox_rois)
                    if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                        bbox_semantic_feat = F.adaptive_avg_pool2d(
                            bbox_semantic_feat, bbox_feats.shape[-2:])
                    bbox_feats += bbox_semantic_feat


        # assign candidate pairs to a relation class
        # (including no-relationship class)
        relations, relation_targets = \
            assign_pairs(sbj_bboxes, sbj_labels, sbj_idxs,
                         obj_bboxes, obj_labels, obj_idxs,
                         gt_bboxes, gt_labels, gt_rel, gt_instid)

        num_relations = relations.shape[0]
        num_relation_cats = 31  # hard coded
        targets = relations.new_zeros((num_relations, num_relation_cats))
        for i in range(num_relations):
            targets[i, relation_targets[i]] = 1

        # get union bboxes
        sbj_bboxes = combined_bboxes[relations[:, -2].long(), :]
        obj_bboxes = combined_bboxes[relations[:, -1].long(), :]
        union_bboxes = bbox_union(sbj_bboxes, obj_bboxes)
        union_rois = bbox2roi([union_bboxes])

        # get visual features
        if use_feat_fusion is True:
            sbj_feats = feats[relations[:, -2].long(), ...]
            obj_feats = feats[relations[:, -1].long(), ...]
        elif use_mask_feat is True:
            sbj_feats = mask_feats[relations[:, -2].long(), ...]
            obj_feats = mask_feats[relations[:, -1].long(), ...]
        else:
            sbj_feats = bbox_feats[relations[:, -2].long(), ...]
            obj_feats = bbox_feats[relations[:, -1].long(), ...]

        with torch.no_grad():
            if use_feat_fusion is True:
                union_feats_mask = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], union_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    union_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], union_rois)
                    if union_semantic_feat.shape[-2:] != union_feats_mask.shape[-2:]:
                        union_semantic_feat = F.adaptive_avg_pool2d(
                            union_semantic_feat, union_feats_mask.shape[-2:])
                    union_feats_mask += union_semantic_feat

                union_feats = bbox_roi_extractor(
                    x[:len(bbox_roi_extractor.featmap_strides)], union_rois)
                if self.with_semantic and 'bbox' in self.semantic_fusion:
                    union_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], union_rois)
                    if union_semantic_feat.shape[-2:] != union_feats.shape[-2:]:
                        union_semantic_feat = F.adaptive_avg_pool2d(
                            union_semantic_feat, union_feats.shape[-2:])
                    union_feats += union_semantic_feat

                union_feats = F.upsample_bilinear(union_feats, None, 2.0)
                if fusion_method == 'max':
                    union_feats = torch.max(union_feats, union_feats_mask)
                else:
                    union_feats = union_feats + union_feats_mask
            elif use_mask_feat is True:
                union_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], union_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    union_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], union_rois)
                    if union_semantic_feat.shape[-2:] != union_feats.shape[-2:]:
                        union_semantic_feat = F.adaptive_avg_pool2d(
                            union_semantic_feat, union_feats.shape[-2:])
                    union_feats += union_semantic_feat
            else:
                union_feats = bbox_roi_extractor(
                    x[:len(bbox_roi_extractor.featmap_strides)], union_rois)
                if self.with_semantic and 'bbox' in self.semantic_fusion:
                    union_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], union_rois)
                    if union_semantic_feat.shape[-2:] != union_feats.shape[-2:]:
                        union_semantic_feat = F.adaptive_avg_pool2d(
                            union_semantic_feat, union_feats.shape[-2:])
                    union_feats += union_semantic_feat

        '''
        ## debug
        if debug_image is not None:
            import os
            import matplotlib.pyplot as plt
            debug_dir = os.path.join('./tmp/{}'.format(debug_filename[:-4]))
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)

            debug_num_relation = relations.shape[0]
            for i in range(debug_num_relation):
                sbj_label = relations[i, 0].to(dtype=torch.int)
                obj_label = relations[i, 1].to(dtype=torch.int)
                rel_label = relations[i, 2].to(dtype=torch.int)
                sbj_index = relations[i, 3].to(dtype=torch.int)
                obj_index = relations[i, 4].to(dtype=torch.int)

                sbj_bbox = combined_bboxes[sbj_index, :4]
                obj_bbox = combined_bboxes[obj_index, :4]

                union_bbox = union_bboxes[i, :4]

                # plot
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(debug_image, cmap=plt.cm.gray)

                x1, y1, x2, y2 = sbj_bbox
                ax.add_artist(plt.Rectangle((x1, y1), x2-x1+1, y2-y1+1,
                                            fill=False, color='r'))
                sbj_text = PicDatasetV20.CATEGORIES[sbj_label-1]['name']
                ax.add_artist(plt.Text(x1, y1, sbj_text, size='x-large',
                                       color='r'))

                x1, y1, x2, y2 = obj_bbox
                ax.add_artist(plt.Rectangle((x1, y1), x2-x1+1, y2-y1+1,
                                            fill=False, color='g'))
                obj_text = PicDatasetV20.CATEGORIES[obj_label-1]['name']
                ax.add_artist(plt.Text(x1, y1, obj_text, size='x-large',
                                       color='g'))

                x1, y1, x2, y2 = union_bbox
                ax.add_artist(plt.Rectangle((x1, y1), x2-x1+1, y2-y1+1,
                                            fill=False, color='w'))
                rel_text = PicDatasetV20.REL_CATEGORIES[rel_label-1]['name'] \
                    if rel_label > 0 else 'None'
                ax.add_artist(plt.Text(x1, y1, rel_text, size='x-large',
                                       color='w',
                                       bbox=dict(facecolor='k', alpha=0.5)))

                rel_text = PicDatasetV20.REL_CATEGORIES[rel_label-1]['name'] \
                    if rel_label > 0 else 'None'
                title = '<{}, {}, {}>'.format(sbj_text, obj_text, rel_text)
                ax.set_title(title)
                ax.axis('off')

                savename = os.path.join(debug_dir, '{:05d}.png'.format(i))
                plt.savefig(savename, dpi=100)
                plt.close()
        '''

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

    '''
    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      proposals=None,
                      gt_instid=None,
                      gt_rel=None):

        with torch.no_grad():
            x = self.extract_feat(img)

        if self.debug is True:
            filename = img_meta[0]['filename']
            debug_image = tensor2imgs(img,
                                      [123.675, 116.28, 103.53],
                                      [58.395, 57.12, 57.375], to_rgb=False)[0]
        else:
            debug_image, filename = None, img_meta[0]['filename']

        rel_binary_loss = \
            self._rel_forward_train_binary(x,
                                           gt_bboxes=gt_bboxes[0],
                                           gt_labels=gt_labels[0],
                                           gt_rel=gt_rel[0],
                                           gt_instid=gt_instid[0],
                                           debug_image=debug_image,
                                           debug_filename=filename)

        losses = dict()
        for name, value in rel_binary_loss.items():
            losses['rel_binary.{}'.format(name)] = (value)

        return losses
    '''

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      proposals=None,
                      gt_instid=None,
                      gt_rel=None):

        use_mask_feat = self.use_mask_feat
        use_feat_fusion = self.use_feat_fusion
        fusion_method = self.fusion_method
        # extract features from detection backbone
        with torch.no_grad():
            x = self.extract_feat(img)

        losses = dict()

        # RPN part, the same as normal two-stage detectors
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

        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses['loss_semantic_seg'] = loss_seg
        else:
            semantic_feat = None

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
            bbox_sampler = build_sampler(rcnn_train_cfg.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            loss_bbox, rois, bbox_targets, bbox_pred = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_targets[0]

            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                # interleaved execution: use regressed bboxes by the box branch
                # to train the mask branch
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(
                            rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []
                        for j in range(num_imgs):
                            assign_result = bbox_assigner.assign(
                                proposal_list[j], gt_bboxes[j],
                                gt_bboxes_ignore[j], gt_labels[j])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_bboxes[j],
                                gt_labels[j],
                                feats=[lvl_feat[j][None] for lvl_feat in x])
                            sampling_results.append(sampling_result)
                loss_mask = self._mask_forward_train(i, x, sampling_results,
                                                     gt_masks, rcnn_train_cfg,
                                                     semantic_feat)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)

        if self.debug is True:
            filename = img_meta[0]['filename']
            debug_image = tensor2imgs(img,
                                      [123.675, 116.28, 103.53],
                                      [58.395, 57.12, 57.375], to_rgb=False)[0]
        else:
            debug_image, filename = None, img_meta[0]['filename']

        batch_size = len(img)
        rel_losses = None
        rank_losses = None
        for i in range(batch_size):

            im_height, im_width, _ = img_meta[i]['img_shape']
            # now only support batch with size 1
            x_i = []
            for xx in x:
                x_i.append(xx[None, i, ...])

            if semantic_feat is not None:
                semantic_feat_i = semantic_feat[None, i, ...]
            else:
                semantic_feat_i = None
            rel_loss = self._rel_forward_train(
                x_i, gt_bboxes=gt_bboxes[i], gt_labels=gt_labels[i],
                gt_rel=gt_rel[i], gt_instid=gt_instid[i],
                rel_train_cfg=self.train_cfg.rel,
                semantic_feat=semantic_feat_i,
                debug_image=debug_image, debug_filename=filename,
                im_width=im_width, im_height=im_height,
                use_mask_feat=use_mask_feat,
                use_feat_fusion=use_feat_fusion,
                fusion_method=fusion_method)

            #rank_loss = self._rel_forward_train_ranking(
            #    x_i, gt_bboxes=gt_bboxes[i], gt_labels=gt_labels[i],
            #    gt_rel=gt_rel[i], gt_instid=gt_instid[i],
            #    im_width=im_width, im_height=im_height,
            #    use_mask_feat=False)

            if rel_losses is None:
                rel_losses = rel_loss
            else:
                for name, value in rel_loss.items():
                    rel_losses[name] += value

            #if rank_losses is None:
            #    rank_losses = rank_loss
            #else:
            #    for name, value in rank_loss.items():
            #        rank_losses[name] += value

        for name, value in rel_losses.items():
            losses['rel.{}'.format(name)] = (value / batch_size)

        #for name, value in rank_losses.items():
        #    losses['rank.{}'.format(name)] = (value / batch_size)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        filename = img_meta[0]['filename']
        if self.save_feat is True:
            P2 = x[0].detach().cpu().numpy()
            P3 = x[1].detach().cpu().numpy()
            P4 = x[2].detach().cpu().numpy()
            P5 = x[3].detach().cpu().numpy()
            P6 = x[4].detach().cpu().numpy()

            p4_folder = os.path.join(self.save_folder, 'P4')
            p5_folder = os.path.join(self.save_folder, 'P5')
            p6_folder = os.path.join(self.save_folder, 'P6')
            if not os.path.exists(p4_folder):
                os.makedirs(p4_folder)
            if not os.path.exists(p5_folder):
                os.makedirs(p5_folder)
            if not os.path.exists(p6_folder):
                os.makedirs(p6_folder)
            np.save(os.path.join(p4_folder, filename + '.npy'), P4)
            np.save(os.path.join(p5_folder, filename + '.npy'), P5)
            np.save(os.path.join(p6_folder, filename + '.npy'), P6)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            ms_sema_result = {}
            ms_sema_result['ensemble'] = \
                self.semantic_head.get_semantic_segm(semantic_pred)
        else:
            semantic_feat = None

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
            bbox_head = self.bbox_head[i]
            cls_score, bbox_pred = self._bbox_forward_test(
                i, x, rois, semantic_feat=semantic_feat)
            ms_scores.append(cls_score)

            if self.test_cfg.keep_all_stages:
                det_bboxes, det_labels = bbox_head.get_det_bboxes(
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=rescale,
                    nms_cfg=rcnn_test_cfg)
                bbox_result = bbox2result(det_bboxes, det_labels,
                                          bbox_head.num_classes)
                ms_bbox_result['stage{}'.format(i)] = bbox_result

                if self.with_mask:
                    mask_head = self.mask_head[i]
                    if det_bboxes.shape[0] == 0:
                        mask_classes = mask_head.num_classes - 1
                        segm_result = [[] for _ in range(mask_classes)]
                    else:
                        _bboxes = (
                            det_bboxes[:, :4] *
                            scale_factor if rescale else det_bboxes)
                        mask_pred = self._mask_forward_test(
                            i, x, _bboxes, semantic_feat=semantic_feat)
                        segm_result = mask_head.get_seg_masks(
                            mask_pred, _bboxes, det_labels, rcnn_test_cfg,
                            ori_shape, scale_factor, rescale)
                    ms_segm_result['stage{}'.format(i)] = segm_result

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_meta[0])

        cls_score = sum(ms_scores) / float(len(ms_scores))
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
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
                last_feat = None
                for i in range(self.num_stages):
                    mask_head = self.mask_head[i]
                    if self.mask_info_flow:
                        mask_pred, last_feat = mask_head(mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)
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
            relation_preds = self._rel_forward_test(x,
                                                    det_bboxes,
                                                    det_labels,
                                                    merged_masks,
                                                    scale_factor,
                                                    ori_shape,
                                                    semantic_feat=semantic_feat,
                                                    im_width=im_width,
                                                    im_height=im_height,
                                                    use_mask_feat=self.use_mask_feat,
                                                    use_feat_fusion=self.use_feat_fusion,
                                                    fusion_method=self.fusion_method)
            if self.rel_save_folder is not None:
                np.save(os.path.join(self.rel_save_folder, filename + '.npy'),
                        relation_preds)

        if not self.test_cfg.keep_all_stages:
            if self.with_mask:
                results = (ms_bbox_result['ensemble'],
                           ms_segm_result['ensemble'])
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

        if self.with_rel:
            return results, relation_preds

        return results

    def aug_test(self, imgs, img_metas, proposals=None, rescale=False):
        """Test with augmentations.
                If rescale is False, then returned bboxes and masks will fit the scale
                of imgs[0].
                """
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

            if self.with_semantic:
                _, semantic_feat = self.semantic_head(x)
            else:
                semantic_feat = None

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_head = self.bbox_head[i]
                cls_score, bbox_pred = self._bbox_forward_test(
                    i, x, rois, semantic_feat=semantic_feat)
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

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head[-1].num_classes - 1)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(self.extract_feats(imgs), img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']

                    if self.with_semantic:
                        _, semantic_feat = self.semantic_head(x)
                    else:
                        semantic_feat = None

                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip)
                    mask_rois = bbox2roi([_bboxes])
                    mask_roi_extractor = self.mask_roi_extractor[-1]
                    mask_feats = mask_roi_extractor(
                        x[:len(mask_roi_extractor.featmap_strides)],
                        mask_rois)
                    if self.with_semantic and 'mask' in self.semantic_fusion:
                        mask_semantic_feat = self.semantic_roi_extractor(
                            [semantic_feat], mask_rois)
                        mask_feats += mask_semantic_feat
                    last_feat = None
                    for i in range(self.num_stages):
                        mask_head = self.mask_head[i]
                        if self.mask_info_flow:
                            mask_pred, last_feat = mask_head(mask_feats, last_feat)
                        else:
                            mask_pred = mask_head(mask_feats)
                        aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg.rcnn)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, det_bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor=1.0, rescale=False)
            return bbox_result, segm_result
        else:
            return bbox_result
