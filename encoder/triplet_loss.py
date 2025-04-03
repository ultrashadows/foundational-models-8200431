import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_dist = torch.norm(anchor - positive, dim=1)
        negative_dist = torch.norm(anchor - negative, dim=1)

        losses = torch.relu(positive_dist - negative_dist + self.margin)
        return losses.mean()
