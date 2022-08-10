""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        # ??? handle 0 was deprecated in dataloader.py
        #     by shuffle in validation
        if self.count == 0:
            assert self.sum == 0
            self.avg == 0.0
        else:
            self.avg = self.sum / self.count


def accuracy_pos_neg(output, target, split_pos_neg=True):
    """Computes the accuracy over the {all,positive and negative classes} predictions for the specified values"""
    accu = {}

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    # TARGET calued to shape of PRED.saved for pos_nev.
    targ = target.reshape(1, -1).expand_as(pred)
    correct = pred.eq(targ)
    accu["all"] = correct[:1].reshape(-1).float().sum(0) * 100.0 / target.size(0)
    assert accu["all"] == accu["all"]

    if split_pos_neg:
        correct_pos = (pred == 1) * (targ == 1)
        correct_neg = (pred == 0) * (targ == 0)
        sum0 = (target == 0).sum(0)
        sum1 = (target == 1).sum(0)

        if sum1 == 0:
            accu["pos"] = torch.Tensor([100.0]).to(target.data.device)
        else:
            accu["pos"] = (
                correct_pos[:1].reshape(-1).float().sum(0)
                * 100.0
                / (target == 1).sum(0)
            )

        if sum0 == 0:
            accu["neg"] = torch.Tensor([100.0]).to(target.data.device)
        else:
            accu["neg"] = (
                correct_neg[:1].reshape(-1).float().sum(0)
                * 100.0
                / (target == 0).sum(0)
            )

        assert accu["pos"] == accu["pos"]
        assert accu["neg"] == accu["neg"]

    return accu


def accuracy_pos_neg_class(output, target, path, split_pos_neg=True):
    """Computes the accuracy over the {all,positive and negative classes} predictions for the specified values"""
    accu = {}

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    # TARGET calued to shape of PRED.saved for pos_nev.
    targ = target.reshape(1, -1).expand_as(pred)
    correct = pred.eq(targ)
    accu["all"] = correct[:1].reshape(-1).float().sum(0) * 100.0 / target.size(0)
    assert accu["all"] == accu["all"]

    if split_pos_neg:
        correct_pos = (pred == 1) * (targ == 1)
        correct_neg = (pred == 0) * (targ == 0)
        sum0 = (target == 0).sum(0)
        sum1 = (target == 1).sum(0)

        if sum1 == 0:
            accu["pos"] = torch.Tensor([100.0]).to(target.data.device)
        else:
            accu["pos"] = (
                correct_pos[:1].reshape(-1).float().sum(0)
                * 100.0
                / (target == 1).sum(0)
            )

        if sum0 == 0:
            accu["neg"] = torch.Tensor([100.0]).to(target.data.device)
        else:
            accu["neg"] = (
                correct_neg[:1].reshape(-1).float().sum(0)
                * 100.0
                / (target == 0).sum(0)
            )

        assert accu["pos"] == accu["pos"]
        assert accu["neg"] == accu["neg"]

    return accu
