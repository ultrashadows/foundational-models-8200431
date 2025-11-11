import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class EncoderWrapper(nn.Module):
    def __init__(self, model_name='./models/albertina-1b5', use_projection=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.use_projection = use_projection

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return F.normalize(cls_embedding, p=2, dim=-1)
