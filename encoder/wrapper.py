import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class EncoderWrapper(nn.Module):
    def __init__(self, model_name='./models/albertina-1b5', pooling_strategy='attention'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy

        hidden_size = self.encoder.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, hidden_size // 4)
        )

        if pooling_strategy == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 4, 1)
            )

        if pooling_strategy == 'multihead':
            self.attention_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 4),
                    nn.GELU(),
                    nn.Linear(hidden_size // 4, 1)
                ) for _ in range(4)
            ])

        self._init_weights()

    def _init_weights(self):
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if hasattr(self, 'attention'):
            for module in self.attention:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)

        last_hidden = outputs.last_hidden_state

        if self.training:
            last_hidden = F.dropout(last_hidden, p=0.05, training=self.training)

        if self.pooling_strategy == 'mean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            pooled = sum_embeddings / sum_mask.clamp(min=1e-9)

        elif self.pooling_strategy == 'attention':
            attention_weights = self.attention(last_hidden)
            attention_weights = attention_weights.squeeze(-1)

            attention_weights = attention_weights.masked_fill(attention_mask == 0, -1e9)
            attention_weights = F.softmax(attention_weights, dim=1)

            attention_weights = F.dropout(attention_weights, p=0.1, training=self.training)

            pooled = torch.sum(last_hidden * attention_weights.unsqueeze(-1), dim=1)

        elif self.pooling_strategy == 'cls':
            pooled = last_hidden[:, 0, :]

        elif self.pooling_strategy == 'max':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            last_hidden = last_hidden * input_mask_expanded
            last_hidden = last_hidden.masked_fill(input_mask_expanded == 0, -1e9)
            pooled = torch.max(last_hidden, dim=1)[0]

        elif self.pooling_strategy == 'weighted_mean':
            all_layers = outputs.hidden_states[-4:]
            weights = F.softmax(torch.randn(len(all_layers), device=last_hidden.device), dim=0)
            weighted_hidden = sum(w * layer for w, layer in zip(weights, all_layers))

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(weighted_hidden.size()).float()
            sum_embeddings = torch.sum(weighted_hidden * input_mask_expanded, dim=1)
            sum_mask = input_mask_expanded.sum(dim=1)
            pooled = sum_embeddings / sum_mask.clamp(min=1e-9)

        elif self.pooling_strategy == 'multihead':
            pooled_list = []
            for head in self.attention_heads:
                attn_weights = F.softmax(head(last_hidden).squeeze(-1).masked_fill(attention_mask == 0, -1e9), dim=1)
                pooled_list.append(torch.sum(last_hidden * attn_weights.unsqueeze(-1), dim=1))
            pooled = torch.cat(pooled_list, dim=1)

        pooled = self.projection(pooled)

        normalized = F.normalize(pooled, p=2, dim=-1, eps=1e-8)

        if self.training:
            noise = torch.randn_like(normalized) * 0.01
            normalized = F.normalize(normalized + noise, p=2, dim=-1, eps=1e-8)

        return normalized
