import os
import json
from torch.utils.data import Dataset


class DatasetProcessor(Dataset):
    def __init__(self, tokenizer, split: str, max_length: int, dataset: str):
        assert split in {"train", "test"}, f"Invalid split: {split}"

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_path = os.path.join("datasets", dataset)

        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            all_samples = json.load(f)

        split_index = int(0.8 * len(all_samples))
        if split == "train":
            self.samples = all_samples[:split_index]
        else:
            self.samples = all_samples[split_index:]

        if split == "train":
            from random import shuffle
            shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        anchor_text = item["question"]
        positive_text = item["positive"]
        negative_text = item["negative"]

        anchor = self.tokenizer(
            anchor_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        positive = self.tokenizer(
            positive_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        negative = self.tokenizer(
            negative_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "anchor": {k: v.squeeze(0) for k, v in anchor.items()},
            "positive": {k: v.squeeze(0) for k, v in positive.items()},
            "negative": {k: v.squeeze(0) for k, v in negative.items()},
        }
