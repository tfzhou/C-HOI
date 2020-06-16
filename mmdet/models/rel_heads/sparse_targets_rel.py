"""
Some functions are adapted from Rowan Zellers:
https://github.com/rowanz/neural-motifs
"""
import torch.nn as nn
import torch
import numpy as np
import logging

from .get_dataset_counts_rel import get_rel_counts

from ..registry import HEADS

logger = logging.getLogger(__name__)


# This module is adapted from Rowan Zellers:
# https://github.com/rowanz/neural-motifs/blob/master/lib/sparse_targets.py
# Modified for this project
@HEADS.register_module
class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self,
                 rel_ann_file=None,
                 num_class=None,
                 num_prd_class=None,
                 must_overlap=True,
                 eps=1e-3,
                 mode='pic'):
        """
        mode: 'pic' or 'hoiw'
        """
        super(FrequencyBias, self).__init__()

        assert mode == 'pic' or mode == 'hoiw' or mode == 'vcoco'

        if mode == 'pic':
            if must_overlap:
                savefile = 'pred_dist_overlap.npz'
            else:
                savefile = 'pred_dist_nonoverlap.npz'
        elif mode == 'hoiw':
            savefile = 'pred_dist_overlap_hoiw.npz'
        elif mode == 'vcoco':
            savefile = 'pred_dist_overlap_vcoco.npz'

        if rel_ann_file is not None and num_class is not None\
                and num_prd_class is not None:
            fg_matrix, bg_matrix = get_rel_counts(rel_ann_file, num_class,
                                                  num_prd_class,
                                                  must_overlap=must_overlap)
            bg_matrix += 1
            fg_matrix[:, :, 0] = bg_matrix

            pred_dist = np.log(fg_matrix / (fg_matrix.sum(2)[:, :, None] + 1e-08) + eps)

            self.num_objs = pred_dist.shape[0]
            pred_dist = torch.FloatTensor(pred_dist).view(-1, pred_dist.shape[2])

            np.savez(savefile, pred_dist=pred_dist, num_objs=self.num_objs)
        else:
            save_data = np.load(savefile)
            pred_dist = save_data['pred_dist']
            self.num_objs = torch.tensor(save_data['num_objs'])
            pred_dist = torch.FloatTensor(pred_dist)

        self.rel_baseline = nn.Embedding(pred_dist.size(0), pred_dist.size(1))
        self.rel_baseline.weight.data = pred_dist
        
        logger.info('Frequency bias tables loaded.')

    def rel_index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2] 
        :return: 
        """
        return self.rel_baseline(labels[:, 0] * self.num_objs + labels[:, 1])
