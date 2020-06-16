import torch
import torch.nn as nn


def accuracy(pred, target, topk=1):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def recall_topk(y_pred, y_true, topk=1):
    scores, pred_label = y_pred.topk(topk, dim=1)
    pred_label = pred_label.t()

    cnt = 0
    for i, lb in enumerate(pred_label[0]):
        if y_true[i, lb] == 1:
            cnt += 1

    acc = cnt * 100.0 / y_pred.size(0)

    return torch.tensor(acc)


class Accuracy(nn.Module):

    def __init__(self, topk=(1, )):
        super().__init__()
        self.topk = topk

    def forward(self, pred, target):
        return accuracy(pred, target, self.topk)
