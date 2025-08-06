import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class EncoderWrapper(nn.Module):
    def __init__(self, model_name='./models/albertina-1b5', pooling_strategy='multihead',
                 projection_dim=None, dropout_rates=(0.3, 0.2), noise_std=0.01):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        self.noise_std = noise_std
        hidden_size = self.encoder.config.hidden_size

        if projection_dim is None:
            projection_dim = hidden_size // 4

        if pooling_strategy == 'multihead':
            self.attention_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 4),
                    nn.GELU(),
                    nn.Linear(hidden_size // 4, 1)
                ) for _ in range(4)
            ])
            projection_input_dim = hidden_size * 4
        elif pooling_strategy == 'mean':
            projection_input_dim = hidden_size
        elif pooling_strategy == 'cls':
            projection_input_dim = hidden_size
        elif pooling_strategy == 'max':
            projection_input_dim = hidden_size
        else:
            raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")

        self.projection = nn.Sequential(
            nn.Linear(projection_input_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, projection_dim)
        )

    def apply_pooling(self, hidden_states, attention_mask):
        if self.pooling_strategy == 'cls':
            return hidden_states[:, 0]

        elif self.pooling_strategy == 'mean':
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            summed = torch.sum(hidden_states * mask, dim=1)
            count = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / count

        elif self.pooling_strategy == 'max':
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states = hidden_states.masked_fill(mask == 0, -1e9)
            return torch.max(hidden_states, dim=1).values

        elif self.pooling_strategy == 'multihead':
            pooled_heads = []
            for head in self.attention_heads:
                attn_weights = head(hidden_states).squeeze(-1)
                attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e9)
                attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)
                weighted = torch.sum(hidden_states * attn_weights, dim=1)
                pooled_heads.append(weighted)
            return torch.cat(pooled_heads, dim=1)

        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        pooled = self.apply_pooling(hidden_states, attention_mask)

        if self.noise_std > 0:
            noise = torch.randn_like(pooled) * self.noise_std
            pooled = pooled + noise

        projected = self.projection(pooled)
        return F.normalize(projected, p=2, dim=-1)
