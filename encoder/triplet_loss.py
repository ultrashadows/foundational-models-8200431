import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3, mining_strategy='batch_hard'):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        self.temperature = 0.1
        self.eps = 1e-8

    def forward(self, anchor, positive, negative, weights=None):
        anchor_norm = F.normalize(anchor, p=2, dim=1, eps=self.eps)
        positive_norm = F.normalize(positive, p=2, dim=1, eps=self.eps)
        negative_norm = F.normalize(negative, p=2, dim=1, eps=self.eps)

        pos_logits = torch.sum(anchor_norm * positive_norm, dim=1) / self.temperature
        neg_logits = torch.sum(anchor_norm * negative_norm, dim=1) / self.temperature

        logits = torch.stack([pos_logits, neg_logits], dim=1)

        soft_labels = torch.tensor([[1.0, 0.0]], device=logits.device).expand(logits.size(0), -1)
        loss = F.kl_div(F.log_softmax(logits, dim=-1), soft_labels, reduction="batchmean")

        return loss

    def batch_hard_mining(self, embeddings, labels):
        batch_size = embeddings.size(0)
        similarity_matrix = torch.matmul(embeddings, embeddings.T)

        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)

        negative_mask = labels.unsqueeze(1) != labels.unsqueeze(0)

        hardest_positive_dist = []
        hardest_negative_dist = []

        for i in range(batch_size):
            pos_sims = similarity_matrix[i][positive_mask[i]]
            neg_sims = similarity_matrix[i][negative_mask[i]]

            if len(pos_sims) > 0:
                hardest_pos = pos_sims.min()
                hardest_positive_dist.append(1 - hardest_pos)
            else:
                hardest_positive_dist.append(torch.tensor(2.0, device=embeddings.device))

            if len(neg_sims) > 0:
                hardest_neg = neg_sims.max()
                hardest_negative_dist.append(1 - hardest_neg)
            else:
                hardest_negative_dist.append(torch.tensor(0.0, device=embeddings.device))

        positive_dist = torch.stack(hardest_positive_dist)
        negative_dist = torch.stack(hardest_negative_dist)

        return positive_dist, negative_dist

    def set_margin(self, new_margin: float):
        self.margin = max(0.05, min(new_margin, 2.0))
