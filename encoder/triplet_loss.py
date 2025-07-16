import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        positive_dist = 1 - F.cosine_similarity(anchor, positive, dim=-1)
        negative_dist = 1 - F.cosine_similarity(anchor, negative, dim=-1)

        losses = F.relu(positive_dist - negative_dist + self.margin)
        return losses.mean()

    def set_margin(self, new_margin: float):
        self.margin = new_margin
