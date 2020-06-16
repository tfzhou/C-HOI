import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ..registry import LOSSES


@LOSSES.register_module
class ReldnBCELoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 reduction='mean',
                 loss_weight=1.0):
        super(ReldnBCELoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        loss_cls = F.binary_cross_entropy(cls_score, label)

        return loss_cls


@LOSSES.register_module
class ReldnContrastiveLoss(nn.Module):

    def __init__(self, node_contrastive_margin=0.2):
        super(ReldnContrastiveLoss, self).__init__()

        self.node_contrastive_margin = node_contrastive_margin

    def forward(self,
                prd_scores_sbj_pos,
                prd_scores_obj_pos,
                rel_ret):
        pass


def reldn_contrastive_losses(prd_scores_sbj_pos, prd_scores_obj_pos, rel_ret, node_contrastive_margin=0.2):
    # sbj
    prd_probs_sbj_pos = F.softmax(prd_scores_sbj_pos, dim=1)
    sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target = split_pos_neg_spo_agnostic(
        prd_probs_sbj_pos, rel_ret['binary_labels_sbj_pos_int32'], rel_ret['inds_unique_sbj_pos'], rel_ret['inds_reverse_sbj_pos'])
    sbj_contrastive_loss = F.margin_ranking_loss(sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target,
                                                 margin=node_contrastive_margin)
    # obj
    prd_probs_obj_pos = F.softmax(prd_scores_obj_pos, dim=1)
    obj_pair_pos_batch, obj_pair_neg_batch, obj_target = split_pos_neg_spo_agnostic(
        prd_probs_obj_pos, rel_ret['binary_labels_obj_pos_int32'], rel_ret['inds_unique_obj_pos'], rel_ret['inds_reverse_obj_pos'])
    obj_contrastive_loss = F.margin_ranking_loss(obj_pair_pos_batch, obj_pair_neg_batch, obj_target,
                                                 margin=node_contrastive_margin)

    return sbj_contrastive_loss, obj_contrastive_loss


def reldn_so_contrastive_losses(prd_scores_sbj_pos, prd_scores_obj_pos, rel_ret, node_contrastive_so_aware_margin=0.2):
    # sbj
    prd_probs_sbj_pos = F.softmax(prd_scores_sbj_pos, dim=1)
    sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target = split_pos_neg_so_aware(
        prd_probs_sbj_pos,
        rel_ret['binary_labels_sbj_pos_int32'], rel_ret['inds_unique_sbj_pos'], rel_ret['inds_reverse_sbj_pos'],
        rel_ret['sbj_labels_sbj_pos_int32'], rel_ret['obj_labels_sbj_pos_int32'], 's')
    sbj_so_contrastive_loss = F.margin_ranking_loss(sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target,
                                                    margin=node_contrastive_so_aware_margin)
    # obj
    prd_probs_obj_pos = F.softmax(prd_scores_obj_pos, dim=1)
    obj_pair_pos_batch, obj_pair_neg_batch, obj_target = split_pos_neg_so_aware(
        prd_probs_obj_pos,
        rel_ret['binary_labels_obj_pos_int32'], rel_ret['inds_unique_obj_pos'], rel_ret['inds_reverse_obj_pos'],
        rel_ret['sbj_labels_obj_pos_int32'], rel_ret['obj_labels_obj_pos_int32'], 'o')
    obj_so_contrastive_loss = F.margin_ranking_loss(obj_pair_pos_batch, obj_pair_neg_batch, obj_target,
                                                    margin=node_contrastive_so_aware_margin)

    return sbj_so_contrastive_loss, obj_so_contrastive_loss


def reldn_p_contrastive_losses(prd_scores_sbj_pos, prd_scores_obj_pos, prd_bias_scores_sbj_pos,
                               prd_bias_scores_obj_pos, rel_ret, node_contrative_p_aware_margin=0.2):
    # sbj
    prd_probs_sbj_pos = F.softmax(prd_scores_sbj_pos, dim=1)
    prd_bias_probs_sbj_pos = F.softmax(prd_bias_scores_sbj_pos, dim=1)
    sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target = split_pos_neg_p_aware(
        prd_probs_sbj_pos,
        prd_bias_probs_sbj_pos,
        rel_ret['binary_labels_sbj_pos_int32'], rel_ret['inds_unique_sbj_pos'], rel_ret['inds_reverse_sbj_pos'],
        rel_ret['prd_labels_sbj_pos_int32'])
    sbj_p_contrastive_loss = F.margin_ranking_loss(sbj_pair_pos_batch, sbj_pair_neg_batch, sbj_target,
                                                   margin=node_contrative_p_aware_margin)
    # obj
    prd_probs_obj_pos = F.softmax(prd_scores_obj_pos, dim=1)
    prd_bias_probs_obj_pos = F.softmax(prd_bias_scores_obj_pos, dim=1)
    obj_pair_pos_batch, obj_pair_neg_batch, obj_target = split_pos_neg_p_aware(
        prd_probs_obj_pos,
        prd_bias_probs_obj_pos,
        rel_ret['binary_labels_obj_pos_int32'], rel_ret['inds_unique_obj_pos'], rel_ret['inds_reverse_obj_pos'],
        rel_ret['prd_labels_obj_pos_int32'])
    obj_p_contrastive_loss = F.margin_ranking_loss(obj_pair_pos_batch, obj_pair_neg_batch, obj_target,
                                                   margin=node_contrative_p_aware_margin)

    return sbj_p_contrastive_loss, obj_p_contrastive_loss


def split_pos_neg_spo_agnostic(prd_probs, binary_labels_pos, inds_unique_pos, inds_reverse_pos):
    device_id = prd_probs.get_device()
    prd_pos_probs = 1 - prd_probs[:, 0]  # shape is (#rels,)
    # loop over each group
    pair_pos_batch = torch.ones(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    pair_neg_batch = torch.zeros(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    for i in range(inds_unique_pos.shape[0]):
        inds = np.where(inds_reverse_pos == i)[0]
        prd_pos_probs_i = prd_pos_probs[inds]
        binary_labels_pos_i = binary_labels_pos[inds]
        pair_pos_inds = np.where(binary_labels_pos_i > 0)[0]
        pair_neg_inds = np.where(binary_labels_pos_i == 0)[0]
        if pair_pos_inds.size == 0 or pair_neg_inds.size == 0:  # ignore this node if either pos or neg does not exist
            continue
        prd_pos_probs_i_pair_pos = prd_pos_probs_i[pair_pos_inds]
        prd_pos_probs_i_pair_neg = prd_pos_probs_i[pair_neg_inds]
        min_prd_pos_probs_i_pair_pos = torch.min(prd_pos_probs_i_pair_pos)
        max_prd_pos_probs_i_pair_neg = torch.max(prd_pos_probs_i_pair_neg)
        pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pair_pos.unsqueeze(0)))
        pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_pair_neg.unsqueeze(0)))

    target = torch.ones_like(pair_pos_batch).cuda(device_id)

    return pair_pos_batch, pair_neg_batch, target


def split_pos_neg_so_aware(prd_probs, binary_labels_pos, inds_unique_pos, inds_reverse_pos,
                           sbj_labels_pos, obj_labels_pos, s_or_o, use_spo_agnostic_compensation):
    device_id = prd_probs.get_device()
    prd_pos_probs = 1 - prd_probs[:, 0]  # shape is (#rels,)
    # loop over each group
    pair_pos_batch = torch.ones(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    pair_neg_batch = torch.zeros(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    for i in range(inds_unique_pos.shape[0]):
        inds = np.where(inds_reverse_pos == i)[0]
        prd_pos_probs_i = prd_pos_probs[inds]
        binary_labels_pos_i = binary_labels_pos[inds]
        sbj_labels_pos_i = sbj_labels_pos[inds]
        obj_labels_pos_i = obj_labels_pos[inds]
        pair_pos_inds = np.where(binary_labels_pos_i > 0)[0]
        pair_neg_inds = np.where(binary_labels_pos_i == 0)[0]
        if pair_pos_inds.size == 0 or pair_neg_inds.size == 0:  # ignore this node if either pos or neg does not exist
            continue
        prd_pos_probs_i_pair_pos = prd_pos_probs_i[pair_pos_inds]
        prd_pos_probs_i_pair_neg = prd_pos_probs_i[pair_neg_inds]
        sbj_labels_i_pair_pos = sbj_labels_pos_i[pair_pos_inds]
        obj_labels_i_pair_pos = obj_labels_pos_i[pair_pos_inds]
        sbj_labels_i_pair_neg = sbj_labels_pos_i[pair_neg_inds]
        obj_labels_i_pair_neg = obj_labels_pos_i[pair_neg_inds]
        max_prd_pos_probs_i_pair_neg = torch.max(prd_pos_probs_i_pair_neg)  # this is fixed for a given i
        if s_or_o == 's':
            # get all unique object labels
            unique_obj_labels, inds_unique_obj_labels, inds_reverse_obj_labels = np.unique(
                obj_labels_i_pair_pos, return_index=True, return_inverse=True, axis=0)
            for j in range(inds_unique_obj_labels.shape[0]):
                # get min pos
                inds_j = np.where(inds_reverse_obj_labels == j)[0]
                prd_pos_probs_i_pos_j = prd_pos_probs_i_pair_pos[inds_j]
                min_prd_pos_probs_i_pos_j = torch.min(prd_pos_probs_i_pos_j)
                # get max neg
                neg_j_inds = np.where(obj_labels_i_pair_neg == unique_obj_labels[j])[0]
                if neg_j_inds.size == 0:
                    if use_spo_agnostic_compensation:
                        pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                        pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_pair_neg.unsqueeze(0)))
                    continue
                prd_pos_probs_i_neg_j = prd_pos_probs_i_pair_neg[neg_j_inds]
                max_prd_pos_probs_i_neg_j = torch.max(prd_pos_probs_i_neg_j)
                pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_neg_j.unsqueeze(0)))
        else:
            # get all unique subject labels
            unique_sbj_labels, inds_unique_sbj_labels, inds_reverse_sbj_labels = np.unique(
                sbj_labels_i_pair_pos, return_index=True, return_inverse=True, axis=0)
            for j in range(inds_unique_sbj_labels.shape[0]):
                # get min pos
                inds_j = np.where(inds_reverse_sbj_labels == j)[0]
                prd_pos_probs_i_pos_j = prd_pos_probs_i_pair_pos[inds_j]
                min_prd_pos_probs_i_pos_j = torch.min(prd_pos_probs_i_pos_j)
                # get max neg
                neg_j_inds = np.where(sbj_labels_i_pair_neg == unique_sbj_labels[j])[0]
                if neg_j_inds.size == 0:
                    if use_spo_agnostic_compensation:
                        pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                        pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_pair_neg.unsqueeze(0)))
                    continue
                prd_pos_probs_i_neg_j = prd_pos_probs_i_pair_neg[neg_j_inds]
                max_prd_pos_probs_i_neg_j = torch.max(prd_pos_probs_i_neg_j)
                pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_neg_j.unsqueeze(0)))

    target = torch.ones_like(pair_pos_batch).cuda(device_id)

    return pair_pos_batch, pair_neg_batch, target


def split_pos_neg_p_aware(prd_probs, prd_bias_probs, binary_labels_pos,
                          inds_unique_pos, inds_reverse_pos, prd_labels_pos,
                          use_spo_agnostic_compensation):
    device_id = prd_probs.get_device()
    prd_pos_probs = 1 - prd_probs[:, 0]  # shape is (#rels,)
    prd_labels_det = prd_probs[:, 1:].argmax(dim=1).data.cpu().numpy() + 1  # prd_probs is a torch.tensor, exlucding background
    # loop over each group
    pair_pos_batch = torch.ones(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    pair_neg_batch = torch.zeros(1).cuda(device_id)  # a dummy sample in the batch in case there is no real sample
    for i in range(inds_unique_pos.shape[0]):
        inds = np.where(inds_reverse_pos == i)[0]
        prd_pos_probs_i = prd_pos_probs[inds]
        prd_labels_pos_i = prd_labels_pos[inds]
        prd_labels_det_i = prd_labels_det[inds]
        binary_labels_pos_i = binary_labels_pos[inds]
        pair_pos_inds = np.where(binary_labels_pos_i > 0)[0]
        pair_neg_inds = np.where(binary_labels_pos_i == 0)[0]
        if pair_pos_inds.size == 0 or pair_neg_inds.size == 0:  # ignore this node if either pos or neg does not exist
            continue
        prd_pos_probs_i_pair_pos = prd_pos_probs_i[pair_pos_inds]
        prd_pos_probs_i_pair_neg = prd_pos_probs_i[pair_neg_inds]
        prd_labels_i_pair_pos = prd_labels_pos_i[pair_pos_inds]
        prd_labels_i_pair_neg = prd_labels_det_i[pair_neg_inds]
        max_prd_pos_probs_i_pair_neg = torch.max(prd_pos_probs_i_pair_neg)  # this is fixed for a given i
        unique_prd_labels, inds_unique_prd_labels, inds_reverse_prd_labels = np.unique(
            prd_labels_i_pair_pos, return_index=True, return_inverse=True, axis=0)
        for j in range(inds_unique_prd_labels.shape[0]):
            # get min pos
            inds_j = np.where(inds_reverse_prd_labels == j)[0]
            prd_pos_probs_i_pos_j = prd_pos_probs_i_pair_pos[inds_j]
            min_prd_pos_probs_i_pos_j = torch.min(prd_pos_probs_i_pos_j)
            # get max neg
            neg_j_inds = np.where(prd_labels_i_pair_neg == unique_prd_labels[j])[0]
            if neg_j_inds.size == 0:
                if use_spo_agnostic_compensation:
                    pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
                    pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_pair_neg.unsqueeze(0)))
                continue
            prd_pos_probs_i_neg_j = prd_pos_probs_i_pair_neg[neg_j_inds]
            max_prd_pos_probs_i_neg_j = torch.max(prd_pos_probs_i_neg_j)
            pair_pos_batch = torch.cat((pair_pos_batch, min_prd_pos_probs_i_pos_j.unsqueeze(0)))
            pair_neg_batch = torch.cat((pair_neg_batch, max_prd_pos_probs_i_neg_j.unsqueeze(0)))

    target = torch.ones_like(pair_pos_batch).cuda(device_id)

    return pair_pos_batch, pair_neg_batch, target
