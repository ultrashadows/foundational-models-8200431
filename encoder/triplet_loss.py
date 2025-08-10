import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(
        self,
        margin: float = 0.2,
        mining_strategy: str = "none",
        margin_scale: float = 1.0,
        temperature: float = 0.07,
        regularization_weight: float = 0.05,
        regularization_mode: str = "proportional",
        min_reg_weight: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.margin = margin
        self.initial_margin = margin
        self.mining_strategy = mining_strategy
        self.margin_scale = margin_scale
        self.temperature = temperature
        self.regularization_weight = regularization_weight
        self.regularization_mode = regularization_mode
        self.min_reg_weight = min_reg_weight
        self.eps = eps

    def set_margin(self, margin: float):
        self.margin = margin

    def set_regularization(
        self,
        weight: float = None,
        mode: str = None,
        min_reg_weight: float = None
    ):
        if weight is not None:
            self.regularization_weight = weight
        if mode is not None:
            self.regularization_mode = mode
        if min_reg_weight is not None:
            self.min_reg_weight = min_reg_weight

    def _effective_reg_weight(self) -> float:
        if self.regularization_weight <= 0.0:
            return 0.0
        if self.regularization_mode == "fixed":
            return max(self.regularization_weight, self.min_reg_weight)
        scale = (self.margin / (self.initial_margin + self.eps))
        eff = self.regularization_weight * scale
        return max(eff, self.min_reg_weight)

    def forward(self, anchor, positive, negative, weights=None):
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negative = F.normalize(negative, p=2, dim=-1)

        pos_sim = torch.sum(anchor * positive, dim=-1)
        neg_sim = torch.sum(anchor * negative, dim=-1)

        margin_term = self.margin_scale * (self.margin + pos_sim - neg_sim)
        triplet_losses = torch.clamp(margin_term, min=0.0)

        if weights is not None:
            triplet_loss = (triplet_losses * weights).mean()
        else:
            triplet_loss = triplet_losses.mean()

        reg_w = self._effective_reg_weight()
        if reg_w > 0.0:
            reg_term = torch.exp(-triplet_loss / (self.margin + self.eps))
            loss = triplet_loss + reg_w * reg_term
        else:
            loss = triplet_loss

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
