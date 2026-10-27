import pytest
import torch


class WhitespaceTokenizer:
    """Stand-in for a Hugging Face tokenizer: one token per whitespace-separated word."""

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def __call__(self, text, padding=None, truncation=True, max_length=8, return_tensors="pt"):
        ids = [hash(w) % 1000 + 1 for w in text.split()][:max_length]
        ids += [0] * (max_length - len(ids))
        mask = [1 if i else 0 for i in ids]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}


@pytest.fixture
def tokenizer():
    return WhitespaceTokenizer()
