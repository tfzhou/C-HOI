import logging

import torch
from torch import nn
from ..registry import HEADS
from mmdet.core import force_fp32
from ..losses import accuracy

logger = logging.getLogger(__name__)


@HEADS.register_module
class ReldnHeadBinary(nn.Module):
    def __init__(self,
                 dim_in,
                 loss_cls=dict(
                     type='ss',
                     use_sigmoid=True,
                     loss_weight=1.0)):
        super(ReldnHeadBinary, self).__init__()

        self.vis_rank_feats = nn.Sequential(
            nn.Linear(dim_in, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.1))

        self.spt_rank_feats = nn.Sequential(
            nn.Linear(28, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1))

        self.proj = nn.Linear(1024+64, 512)

        self.readout = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.loss_rank = nn.MarginRankingLoss(margin=0.2)
        self.loss_cosine = nn.CosineEmbeddingLoss(margin=0.5)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, vis_feat, spt_feat):
        f_vis = self.vis_rank_feats(vis_feat)
        f_spt = self.spt_rank_feats(spt_feat)

        embedding = torch.cat([f_vis, f_spt], dim=1)

        embedding_proj = self.proj(embedding)

        scores = self.readout(embedding_proj)

        return scores, embedding_proj

    @force_fp32(apply_to=('rel_ranking_score',))
    def loss(self, scores_pos, embedding_pos,
             scores_neg, embedding_neg,
             targets_ranking, targets_cosine):

        bs = embedding_neg.shape[0]

        loss_rank = self.loss_rank(scores_pos, scores_neg, targets_ranking)
        loss_cosine = self.loss_cosine(embedding_pos, embedding_neg,
                                       targets_cosine)
        reg1 = torch.norm(embedding_pos, p=2) / bs
        reg2 = torch.norm(embedding_neg, p=2) / bs

        gamma = 1e-4

        losses = {
            'loss': loss_rank + loss_cosine + gamma * reg1 + gamma * reg2
        }
        #pred = pred.view(-1, 1)
        #labels = labels.view(-1, 1).long()

        #num_pos = torch.sum(labels)
        #num_neg = torch.sum(1 - labels)
        #num_total = num_neg + num_pos

        #pos_weight = num_neg.to(dtype=torch.float) / num_total.to(dtype=torch.float)
        #neg_weight = num_pos.to(dtype=torch.float) / num_total.to(dtype=torch.float)

        #label_weight = labels.new_zeros((num_total, 1), dtype=torch.float)
        #for i in range(num_total):
        #    if labels[i] == 0:
        #        label_weight[i] = neg_weight
        #    else:
        #        label_weight[i] = pos_weight

        #losses = {
        #    'loss': self.loss_cls(pred, labels, weight=label_weight),
        #    'accuracy': accuracy(pred, labels)
        #}
        return losses

