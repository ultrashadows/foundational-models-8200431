import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self,
                 margin=0.3,
                 mining_strategy='adaptive',
                 use_soft_margin=False,
                 margin_scale=5.0,
                 temperature=0.1,
                 regularization_weight=0.1,
                 enable_regularization=True):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        self.use_soft_margin = use_soft_margin
        self.margin_scale = margin_scale
        self.temperature = temperature
        self.regularization_weight = regularization_weight
        self.enable_regularization = enable_regularization
        self.eps = 1e-8

    def forward(self, anchor, positive, negative, weights=None):
        anchor = F.normalize(anchor, p=2, dim=1, eps=self.eps)
        positive = F.normalize(positive, p=2, dim=1, eps=self.eps)
        negative = F.normalize(negative, p=2, dim=1, eps=self.eps)

        pos_sim = torch.sum(anchor * positive, dim=1).clamp(-1 + self.eps, 1 - self.eps)
        neg_sim = torch.sum(anchor * negative, dim=1).clamp(-1 + self.eps, 1 - self.eps)

        pos_dist = 1.0 - pos_sim
        neg_dist = 1.0 - neg_sim

        if self.mining_strategy == 'adaptive':
            margin = self.margin * (1.0 + torch.exp(-self.margin_scale * (pos_sim - neg_sim)))
            margin = margin.clamp(min=0.05, max=2.0)
        else:
            margin = self.margin

        if self.use_soft_margin:
            triplet_scores = pos_dist - neg_dist
            losses = F.softplus(triplet_scores + margin)
        else:
            losses = F.relu(pos_dist - neg_dist + margin)

        if weights is not None:
            losses = losses * weights

        loss = losses.mean()

        if self.enable_regularization:
            loss = loss + self.regularization_weight * torch.exp(-loss / (self.margin + self.eps))

        return loss

    def forward_info_nce(self, anchor, positive):
        anchor = F.normalize(anchor, dim=1, eps=self.eps)
        positive = F.normalize(positive, dim=1, eps=self.eps)
        logits = torch.matmul(anchor, positive.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def forward_softmax_triplet(self, anchor, positive, negative):
        anchor = F.normalize(anchor, p=2, dim=1, eps=self.eps)
        positive = F.normalize(positive, p=2, dim=1, eps=self.eps)
        negative = F.normalize(negative, p=2, dim=1, eps=self.eps)

        pos_scores = torch.sum(anchor * positive, dim=1) / self.temperature
        neg_scores = torch.sum(anchor * negative, dim=1) / self.temperature

        logits = torch.stack([pos_scores, neg_scores], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)

        return F.cross_entropy(logits, labels)

    def batch_hard_mining(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1, eps=self.eps)
        sim_matrix = torch.matmul(embeddings, embeddings.T)

        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        negative_mask = ~positive_mask

        hardest_positive_dist = []
        hardest_negative_dist = []

        for i in range(embeddings.size(0)):
            pos_sims = sim_matrix[i][positive_mask[i]]
            neg_sims = sim_matrix[i][negative_mask[i]]

            hardest_pos = pos_sims.min() if pos_sims.numel() > 0 else torch.tensor(2.0, device=embeddings.device)
            hardest_neg = neg_sims.max() if neg_sims.numel() > 0 else torch.tensor(-1.0, device=embeddings.device)

            hardest_positive_dist.append(1.0 - hardest_pos)
            hardest_negative_dist.append(1.0 - hardest_neg)

        pos_d = torch.stack(hardest_positive_dist)
        neg_d = torch.stack(hardest_negative_dist)
        return pos_d, neg_d

    def set_margin(self, new_margin):
        self.margin = max(0.05, min(new_margin, 2.0))
