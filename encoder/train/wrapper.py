import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class EncoderWrapper(nn.Module):
    """CLS-pooled, L2-normalised encoder with a ``model(input_ids, attention_mask)`` signature.

    Only needed by ``DatasetProcessor.precompute_hard_negatives``, which is disabled in the final
    training configuration. Note that fine-tuning itself uses mean pooling, not CLS pooling.
    """

    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return F.normalize(cls_embedding, p=2, dim=-1)
