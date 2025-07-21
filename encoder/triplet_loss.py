import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3, mining_strategy='adaptive', temperature=0.1):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        self.temperature = temperature
        self.eps = 1e-8

    def forward(self, anchor, positive, negative, weights=None):
        anchor = F.normalize(anchor, p=2, dim=1, eps=self.eps)
        positive = F.normalize(positive, p=2, dim=1, eps=self.eps)
        negative = F.normalize(negative, p=2, dim=1, eps=self.eps)

        pos_sim = torch.sum(anchor * positive, dim=1)
        neg_sim = torch.sum(anchor * negative, dim=1)

        pos_sim = torch.clamp(pos_sim, -1 + self.eps, 1 - self.eps)
        neg_sim = torch.clamp(neg_sim, -1 + self.eps, 1 - self.eps)

        pos_dist = 1 - pos_sim
        neg_dist = 1 - neg_sim

        if self.mining_strategy == 'adaptive':
            margin = self.margin * (1 + torch.exp(-5 * (pos_sim - neg_sim)))
        else:
            margin = self.margin

        losses = F.relu(pos_dist - neg_dist + margin)

        if weights is not None:
            losses = losses * weights

        loss = losses.mean() + 0.1 * torch.exp(-losses.mean() / 0.1)
        return loss

    def forward_info_nce(self, anchor, positive):
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        logits = torch.matmul(anchor, positive.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def batch_hard_mining(self, embeddings, labels):
        sim_matrix = torch.matmul(embeddings, embeddings.T)
        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        negative_mask = labels.unsqueeze(1) != labels.unsqueeze(0)

        hardest_positive_dist = []
        hardest_negative_dist = []

        for i in range(embeddings.size(0)):
            pos_sims = sim_matrix[i][positive_mask[i]]
            neg_sims = sim_matrix[i][negative_mask[i]]

            hardest_pos = pos_sims.min() if len(pos_sims) > 0 else torch.tensor(2.0, device=embeddings.device)
            hardest_neg = neg_sims.max() if len(neg_sims) > 0 else torch.tensor(0.0, device=embeddings.device)

            hardest_positive_dist.append(1 - hardest_pos)
            hardest_negative_dist.append(1 - hardest_neg)

        pos_d = torch.stack(hardest_positive_dist)
        neg_d = torch.stack(hardest_negative_dist)
        return pos_d, neg_d

    def set_margin(self, new_margin):
        self.margin = max(0.05, min(new_margin, 2.0))
