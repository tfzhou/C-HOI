from torch import nn
from mmdet.core import (bbox2result, bbox_mapping)
from mmdet.core import (bbox2roi, merge_aug_masks, merge_aug_bboxes,
                        multiclass_nms, merge_aug_proposals)
from mmdet.models.detectors import BaseDetector


class EnsembleCascadeRCNN(BaseDetector):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def simple_test(self, img, img_meta, rescale=False, **kwargs):
        return self.aug_test([img], [img_meta], rescale=rescale)

    def forward_train(self, imgs, img_metas, **kwargs):
        pass

    def extract_feat(self, imgs):
        pass

    def aug_test(self, imgs, img_metas, rescale=False, **kwargs):
        """
        Test with augmentations.
        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """

        ms_bbox_result = {}

        rpn_test_cfg = self.models[0].test_cfg.rpn
        rcnn_test_cfg = self.models[0].test_cfg.rcnn

        # For each model, compute detections
        aug_bboxes = []
        aug_scores = []
        aug_img_metas = []
        for model in self.models:
            for x, img_meta in zip(model.extract_feats(imgs), img_metas):
                proposal_list = model.simple_test_rpn(x, img_meta, rpn_test_cfg)

                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']

                ms_scores = []
                rois = bbox2roi(proposal_list)
                for i in range(model.num_stages):
                    bbox_head = model.bbox_head[i]
                    bbox_roi_extractor = model.bbox_roi_extractor[i]
                    bbox_feats = bbox_roi_extractor(
                        x[:len(bbox_roi_extractor.featmap_strides)], rois)
                    cls_score, bbox_pred = bbox_head(bbox_feats)
                    ms_scores.append(cls_score)

                    if i < model.num_stages - 1:
                        bbox_label = cls_score.argmax(dim=1)
                        rois = bbox_head.regress_by_class(rois, bbox_label,
                                                          bbox_pred,
                                                          img_meta[0])

                cls_score = sum(ms_scores) / float(len(ms_scores))
                bboxes, scores = model.bbox_head[-1].get_det_bboxes(
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=False,
                    cfg=None)
                aug_bboxes.append(bboxes)
                aug_scores.append(scores)
                aug_img_metas.append(img_meta)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, aug_img_metas, rcnn_test_cfg,
            type='concat')
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes, merged_scores, rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.models[0].bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result

        ori_shape = img_metas[0][0]['ori_shape']
        scale_factor = img_metas[0][0]['scale_factor']

        ensemble_relation_preds = {}
        for model in self.models:
            for x, img_meta in zip(model.extract_feats(imgs), img_metas):
                im_height, im_width, _ = img_meta[0]['img_shape']
                filename = img_meta[0]['filename']
                relation_preds = model._rel_forward_test(x,
                                                        det_bboxes,
                                                        det_labels,
                                                        scale_factor,
                                                        ori_shape,
                                                        im_width=im_width,
                                                        im_height=im_height)
                if filename not in ensemble_relation_preds:
                    ensemble_relation_preds = relation_preds
                    ensemble_relation_preds['file_name'] = filename
                else:
                    ensemble_relation_preds['hoi_prediction'].extend(
                        relation_preds['hoi_prediction'])

        ensemble_relation_preds_remove_dup = ensemble_relation_preds.copy()
        for i, hoi_pred_i in enumerate(ensemble_relation_preds['hoi_prediction']):
            for j, hoi_pred_j in enumerate(ensemble_relation_preds['hoi_prediction']):
                if i != j:
                    sbj_i = hoi_pred_i['subject_id']
                    obj_i = hoi_pred_i['object_id']
                    cat_i = hoi_pred_i['category_id']
                    sbj_j = hoi_pred_j['subject_id']
                    obj_j = hoi_pred_j['object_id']
                    cat_j = hoi_pred_j['category_id']
                    if sbj_i == sbj_j and obj_i == obj_j and cat_i == cat_j:
                        ensemble_relation_preds_remove_dup.remove(hoi_pred_j)

        results = (ms_bbox_result['ensemble'],
                   ensemble_relation_preds_remove_dup)

        return results
