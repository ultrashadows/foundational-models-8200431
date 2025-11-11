import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, base_margin=0.06, distance='cosine', reduction='mean'):
        super().__init__()
        self.base_margin = base_margin
        self.distance = distance
        self.reduction = reduction

    def forward(self, anchor, positive, negative, sample_weights=None):
        if self.distance == 'cosine':
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        elif self.distance == 'euclidean':
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance == 'dot_product':
            pos_dist = -torch.sum(anchor * positive, dim=1)
            neg_dist = -torch.sum(anchor * negative, dim=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")

        current_margins = neg_dist - pos_dist

        adaptive_margins = self.base_margin * torch.sigmoid(current_margins * 10)

        loss = F.relu(pos_dist - neg_dist + adaptive_margins)

        if sample_weights is not None:
            loss = loss * sample_weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
