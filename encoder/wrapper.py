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
        else:
            projection_input_dim = hidden_size

        self.projection = nn.Sequential(
            nn.Linear(projection_input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(hidden_size // 2, projection_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if hasattr(self, 'attention_heads'):
            for head in self.attention_heads:
                for module in head:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

    def _apply_attention_pooling(self, last_hidden, attention_mask):
        pooled_outputs = []
        for head in self.attention_heads:
            attn_weights = head(last_hidden).squeeze(-1)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e9)
            attn_weights = F.softmax(attn_weights, dim=1)
            pooled = torch.sum(last_hidden * attn_weights.unsqueeze(-1), dim=1)
            pooled_outputs.append(pooled)
        return torch.cat(pooled_outputs, dim=-1)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state.to(torch.float32)

        if self.pooling_strategy == 'multihead':
            pooled = self._apply_attention_pooling(last_hidden, attention_mask)
        elif self.pooling_strategy == 'mean':
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            pooled = torch.sum(last_hidden * mask_expanded, dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        elif self.pooling_strategy == 'cls':
            pooled = last_hidden[:, 0, :]
        elif self.pooling_strategy == 'max':
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            last_hidden = last_hidden * mask_expanded
            last_hidden = last_hidden.masked_fill(mask_expanded == 0, -1e9)
            pooled = torch.max(last_hidden, dim=1)[0]
        elif self.pooling_strategy == 'attn_mean':
            mean_pool = torch.sum(last_hidden * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9).unsqueeze(-1)
            attn_pool = self._apply_attention_pooling(last_hidden, attention_mask)
            pooled = torch.cat([mean_pool, attn_pool], dim=-1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        pooled = pooled.to(self.encoder.device)
        pooled = self.projection(pooled)
        normalized = F.normalize(pooled, p=2, dim=-1, eps=1e-8)

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(normalized) * self.noise_std
            normalized = F.normalize(normalized + noise, p=2, dim=-1, eps=1e-8)

        return normalized
