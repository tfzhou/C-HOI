from functools import reduce
import numpy as np
import sys


def compute_iou(target_mask, query_masks):
    N = query_masks.shape[0]
    target_masks = np.repeat(target_mask[None], N, axis=0)
    target_masks = target_masks.astype(np.int32)
    query_masks = query_masks.astype(np.int32)
    I = target_masks & query_masks
    I = I.sum(axis=2).sum(axis=1)
    U = target_masks | query_masks
    U = U.sum(axis=2).sum(axis=1) + sys.float_info.min
    return I/U


def intersect_2d(x1, x2):

    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res


def triplet(rels, semantic, instance):
    subs = rels[:, 0]
    objs = rels[:, 1]
    rel_cats = rels[:, 2]
    subs_mask = instance == subs[:, None, None]
    objs_mask = instance == objs[:, None, None]
    subs_semantic = subs_mask * semantic
    objs_semantic = objs_mask * semantic
    subs_class = subs_semantic.reshape(subs_semantic.shape[0], -1).max(axis=1)
    objs_class = objs_semantic.reshape(objs_semantic.shape[0], -1).max(axis=1)
    triplet_rels = np.concatenate((subs_class[:, None], objs_class[:, None], rel_cats[:, None]), axis=1)
    triplet_masks = np.concatenate((subs_mask[:, None, :, :], objs_mask[:, None, :, :]), axis=1)
    return triplet_rels, triplet_masks


def evaluate_from_dict(gt_entry, pred_entry, result_dict, iou_threshes, rel_cats, geometric_rel_cats):

    gt_rels = gt_entry['relations']
    gt_semantic = gt_entry['semantic']
    gt_instance = gt_entry['instance']
    gt_rels_nums = [0 for x in range(len(rel_cats))]
    for rel in gt_rels:
        gt_rels_nums[rel[2]-1] += 1
        if rel[2] in geometric_rel_cats.keys():
            gt_rels_nums[-2] += 1
        else:
            gt_rels_nums[-1] += 1

    pred_rels = pred_entry['relations']
    pred_semantic = pred_entry['semantic']
    pred_instance = pred_entry['instance']

    gt_triplet_rels, gt_triplet_masks = triplet(gt_rels, gt_semantic, gt_instance)
    pred_triplet_rels, pred_triplet_masks = triplet(pred_rels, pred_semantic, pred_instance)
    keeps = intersect_2d(gt_triplet_rels, pred_triplet_rels)

    gt_has_match = keeps.any(1)
    pred_to_gt = {}
    for iou_thresh in iou_threshes:
        pred_to_gt[iou_thresh] = {}
        for rel_cat_id, rel_cat_name in rel_cats.items():
            pred_to_gt[iou_thresh][rel_cat_name] = [[] for x in range(pred_rels.shape[0])]
    for gt_ind, gt_mask, keep_inds in zip(np.where(gt_has_match)[0], gt_triplet_masks[gt_has_match], keeps[gt_has_match]):
        masks = pred_triplet_masks[keep_inds]
        sub_iou = compute_iou(gt_mask[0], masks[:, 0])
        obj_iou = compute_iou(gt_mask[1], masks[:, 1])
        for iou_thresh in iou_threshes:
            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)
            for i in np.where(keep_inds)[0][inds]:
                for rel_cat_id, rel_cat_name in rel_cats.items():
                    if gt_triplet_rels[int(gt_ind), 2] == rel_cat_id:
                        pred_to_gt[iou_thresh][rel_cat_name][i].append(int(gt_ind))
                if gt_triplet_rels[int(gt_ind), 2] in geometric_rel_cats.keys():
                    pred_to_gt[iou_thresh]['geometric_rel'][i].append(int(gt_ind))
                else:
                    pred_to_gt[iou_thresh]['non_geometric_rel'][i].append(int(gt_ind))
    for iou_thresh in iou_threshes:
        for rel_cat_id, rel_cat_name in rel_cats.items():
            match = reduce(np.union1d, pred_to_gt[iou_thresh][rel_cat_name])
            if gt_rels_nums[rel_cat_id-1] == 0:
                # None means this rel_cat_id does not appear in gt_rels in this img
                rec_i = None
            else:
                rec_i = float(len(match)) / float(gt_rels_nums[rel_cat_id-1])
            result_dict[iou_thresh][rel_cat_name].append(rec_i)




