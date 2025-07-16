import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class EncoderWrapper(nn.Module):
    def __init__(self, model_name='./models/albertina-1b5'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()

        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        mean_pooled = sum_embeddings / sum_mask.clamp(min=1e-9)

        return F.normalize(mean_pooled, p=2, dim=-1)
