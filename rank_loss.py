import torch
import torch.nn as nn
import torch.nn.functional as F


class RankLoss(nn.Module):
    def __init__(self, p=1, eps=1e-6):
        super(RankLoss, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x1, x2, t):
        diff = F.pairwise_distance(x1, x2, self.p, self.eps)
        loss = (1 - t) * diff / 2 + torch.log(1 + torch.exp(-diff))
        return loss
