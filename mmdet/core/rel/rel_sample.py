import torch
import numpy as np

from mmdet.core import bbox_overlaps, bbox2delta, bbox2c, bbox_union

PICV2_DEFINED_RELATIONS = 31


def get_spatial_feature(bs, bo, im_width, im_height):
    bu = bbox_union(bs, bo)

    spt_feat_so = get_pair_feature(bs, bo)
    spt_feat_su = get_pair_feature(bs, bu)
    spt_feat_uo = get_pair_feature(bu, bo)

    spt_feat_bs = bbox2c(bs, im_width, im_height)
    spt_feat_bo = bbox2c(bo, im_width, im_height)

    spt_feat = torch.cat([spt_feat_bs, spt_feat_bo, spt_feat_so,
                          spt_feat_su, spt_feat_uo], dim=-1)  # 28-D

    return spt_feat


def get_pair_feature(bs, bo):
    delta1 = bbox2delta(bo, bs)
    delta2 = bbox2delta(bs, bo)
    spt_feat = torch.cat([delta1, delta2[:, :2]], dim=-1)
    return spt_feat


def assign_pairs(sbj_bboxes, sbj_labels, sbj_idxs,
                 obj_bboxes, obj_labels, obj_idxs,
                 gt_bboxes, gt_labels, gt_rel, gt_instid,
                 th=0.5):
    """

    :param sbj_bboxes: m x 4 (tensor)
    :param sbj_labels: m x 1 (tensor)
    :param sbj_idxs:   m x 1 (tensor) subject indices in combined_bboxes
    :param obj_bboxes: m x 4 (tensor)
    :param obj_labels: m x 1 (tensor)
    :param obj_idxs:   m x 1 (tensor) object indices in combined_bboxes
    :param gt_bboxes:  n x 4 (tensor)
    :param gt_labels:  n x 1 (tensor)
    :param gt_rel:     k x 3 (tensor) [subject instance id, object instance id, relation id]
    :param gt_instid:  n x 1 (tensor) instance indices
    :param th:         IoU threshold (default: 0.5)
    :return:
        positive and negative quintet,
        each quad: [subject semantic id, object semantic id, relation id,
                    subject index, object index ]
    """

    total_cand = sbj_bboxes.shape[0]
    total_rela = gt_rel.shape[0]

    # return variables
    relations = []
    relation_labels = []

    # compute overlaps
    overlaps_sbj_gt = bbox_overlaps(sbj_bboxes, gt_bboxes)
    overlaps_obj_gt = bbox_overlaps(obj_bboxes, gt_bboxes)

    # convert to numpy
    for i in range(total_cand):
        # labels for detected subject and object
        sbj_label = sbj_labels[i][0]
        obj_label = obj_labels[i][0]
        found = False

        r = [sbj_label, obj_label, sbj_idxs[i], obj_idxs[i]]
        r_l = []
        for j in range(total_rela):
            # relations
            gt_sbj_instid, gt_obj_instid, gt_rel_id = gt_rel[j, :]
            sbj_instid_idx = (gt_instid == gt_sbj_instid.item()).nonzero()[0]
            obj_instid_idx = (gt_instid == gt_obj_instid.item()).nonzero()[0]

            assert len(sbj_instid_idx) == 1
            assert len(obj_instid_idx) == 1

            gt_sbj_label = gt_labels[sbj_instid_idx][0]
            gt_obj_label = gt_labels[obj_instid_idx][0]

            overlap_s = overlaps_sbj_gt[i, sbj_instid_idx]
            overlap_o = overlaps_obj_gt[i, obj_instid_idx]

            positive = (sbj_label.item() == gt_sbj_label.item() and
                        obj_label.item() == gt_obj_label.item() and
                        overlap_s.item() >= th and overlap_o.item() >= th)

            if positive is True:
                r_l.append(gt_rel_id)
                found = True
        if not found:
            relation_labels.append(torch.tensor([0]))
        else:
            relation_labels.append(r_l)
        relations.append(r)

    assert len(relations) == len(relation_labels)

    device = torch.get_device(sbj_bboxes)
    relations = torch.FloatTensor(relations).to(device)

    return relations, relation_labels


def sample_pairs(det_bboxes, det_labels, overlap=False, overlap_th=-1,
                 test=False, track='pic'):
    """Sample candidate subject-subject and subject-object pairs

    Args
        det_bboxes (Tensor): shape (n, 4)
        det_labels (Tensor): shape (n, 1)

    Returns
        subject_bboxes (Tensor): shape (m, 4), m is the total number of sampling pairs
        subject_labels (Tensor): shape (m, 1)
        object_bboxes (Tensor): shape (m, 4)
        object_labels (Tensor): shape (m, 1)
    """

    assert det_bboxes.shape[0] == det_labels.shape[0]
    assert overlap is False or (overlap is True and overlap_th > 0)

    subject_idx = (det_labels == 1).nonzero().squeeze(1)
    object_idx = (det_labels > 1).nonzero().squeeze(1)

    if len(subject_idx) == 0:
        return None, None, None, None, None, None

    subject_bboxes = torch.index_select(det_bboxes, 0, subject_idx)
    subject_labels = torch.index_select(det_labels, 0, subject_idx)

    object_bboxes = torch.index_select(det_bboxes, 0, object_idx)
    object_labels = torch.index_select(det_labels, 0, object_idx)

    # in order to build subject-subject pairs
    object_bboxes = torch.cat((subject_bboxes, object_bboxes), dim=0)
    object_labels = torch.cat((subject_labels, object_labels), dim=0)
    object_idx = torch.cat((subject_idx, object_idx), dim=0)

    num_subject = subject_bboxes.shape[0]
    num_object = object_bboxes.shape[0]

    subject_bboxes = subject_bboxes.repeat_interleave(num_object, dim=0)
    subject_labels = subject_labels.repeat_interleave(num_object, dim=0)
    subject_idx = subject_idx.repeat_interleave(num_object, dim=0)

    object_bboxes = object_bboxes.repeat(num_subject, 1)
    object_labels = object_labels.repeat(1, num_subject)
    object_idx = object_idx.repeat(1, num_subject).squeeze(0)

    subject_labels = subject_labels.view(-1, 1)
    object_labels = object_labels.view(-1, 1)

    assert len(subject_idx) == len(object_idx),\
        '{} {}'.format(len(subject_idx), len(object_idx))
    assert subject_bboxes.shape == object_bboxes.shape,\
        '{} {}'.format(subject_bboxes.shape, object_bboxes.shape)
    assert subject_labels.shape == object_labels.shape,\
        '{} {}'.format(subject_labels.shape, object_labels.shape)

    # remove self pairs
    mask = np.ones(subject_bboxes.shape[0], dtype=bool)
    for i in range(num_subject):
        mask[i + num_object * i] = False

    keep = np.where(mask)[0]
    subject_bboxes = subject_bboxes[keep, :]
    subject_labels = subject_labels[keep]
    subject_idx = subject_idx[keep]
    object_bboxes = object_bboxes[keep, :]
    object_labels = object_labels[keep]
    object_idx = object_idx[keep]

    if overlap is True and subject_bboxes.shape[0] > 0 and\
            test is True and track is 'hoiw':
        overlaps_iob = bbox_overlaps(subject_bboxes[:, :4],
                                     object_bboxes[:, :4],
                                     mode='iob', is_aligned=True)

        overlaps_iof = bbox_overlaps(subject_bboxes[:, :4],
                                     object_bboxes[:, :4],
                                     mode='iof', is_aligned=True)

        keep = []
        overlaps_iob = list(overlaps_iob.cpu().numpy())
        overlaps_iof = list(overlaps_iof.cpu().numpy())
        object_labels_list = list(object_labels.cpu().numpy())
        for i, (iob, iof, obj_lb) in\
                enumerate(zip(overlaps_iob, overlaps_iof, object_labels_list)):
            if iob > overlap_th or iof > overlap_th:
                keep.append(i)

        subject_bboxes = subject_bboxes[keep, :]
        subject_labels = subject_labels[keep]
        subject_idx = subject_idx[keep]
        object_bboxes = object_bboxes[keep, :]
        object_labels = object_labels[keep]
        object_idx = object_idx[keep]

    return subject_bboxes, subject_labels, subject_idx, object_bboxes,\
           object_labels, object_idx
