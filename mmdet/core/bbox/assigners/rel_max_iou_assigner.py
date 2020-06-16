import torch

from ..geometry import bbox_overlaps
from .base_assigner import BaseAssigner


class RelMaxIoUAssigner(BaseAssigner):
    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 ):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou

    def assign(self,
               sbj_bboxes, sbj_labels,
               obj_bboxes, obj_labels,
               gt_labels_rel, gt_bboxes_sbj, gt_bboxes_obj):

        assert sbj_bboxes.shape[0] == sbj_labels.shape[0]
        assert obj_bboxes.shape[0] == obj_labels.shape[0]
        assert sbj_labels.shape[0] == obj_labels.shape[0]
        assert gt_labels_rel.shape[0] == gt_bboxes_sbj.shape[0] == gt_bboxes_obj.shape[0]

        m = sbj_labels.shape[0]
        n = gt_labels_rel.shape[0]

        sbj_bboxes_ = sbj_bboxes[:, :4]
        gt_bboxes_sbj_ = gt_bboxes_sbj[:, :4]
        overlaps_sbj = bbox_overlaps(gt_bboxes_sbj_, sbj_bboxes_)

        obj_bboxes_ = obj_bboxes[:, :4]
        gt_bboxes_obj_ = gt_bboxes_obj[:, :4]
        overlaps_obj = bbox_overlaps(gt_bboxes_obj_, obj_bboxes_)

        assigned_labels = []
        for i in range(m):
            overlap_i = (overlaps_sbj[i, :] + overlaps_obj[i, :]) / 2
            max_idx = torch.argmax(overlap_i)

            sbj_overlap = overlaps_sbj[i, max_idx]
            obj_overlap = overlaps_obj[i, max_idx]

            if sbj_overlap > self.pos_iou_thr and obj_overlap > self.pos_iou_thr:
                assigned_labels.append(gt_labels_rel[max_idx])

        return assigned_labels









