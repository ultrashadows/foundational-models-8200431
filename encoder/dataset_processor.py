import os
import json
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class DatasetProcessor(Dataset):
    def __init__(self, tokenizer, split: str, max_length: int, dataset: str, hard_negatives=True):
        assert split in {"train", "test"}, f"Invalid split: {split}"

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_path = os.path.join("datasets", dataset)
        self.hard_negatives = hard_negatives

        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            all_samples = json.load(f)

        split_index = int(0.8 * len(all_samples))
        if split == "train":
            self.samples = all_samples[:split_index]
        else:
            self.samples = all_samples[split_index:]

        self.negative_pool = []
        for sample in self.samples:
            if "negative" in sample:
                self.negative_pool.append(sample["negative"])
            if "positive" in sample:
                self.negative_pool.append(sample["positive"])

        self.hard_negative_texts = None
        self.semi_hard_negative_texts = None
        self.difficulty_weights = None

        if split == "train":
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        anchor_text = item["question"]
        positive_text = item["positive"]

        negative_selection_strategy = random.random()

        if self.hard_negatives and self.hard_negative_texts is not None and negative_selection_strategy < 0.4:
            negative_text = self.hard_negative_texts[idx]
        elif self.hard_negatives and self.semi_hard_negative_texts is not None and negative_selection_strategy < 0.7:
            negative_text = self.semi_hard_negative_texts[idx]
        elif self.hard_negatives and negative_selection_strategy < 0.85:
            negative_text = random.choice(self.negative_pool)
        else:
            negative_text = item["negative"]

        if random.random() < 0.05:
            anchor_text = self._augment_text(anchor_text)
        if random.random() < 0.05:
            positive_text = self._augment_text(positive_text)
        if random.random() < 0.03:
            negative_text = self._augment_text(negative_text)

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

        sample_weight = 1.0
        if self.difficulty_weights is not None:
            sample_weight = self.difficulty_weights[idx]

        return {
            "anchor": {k: v.squeeze(0) for k, v in anchor.items()},
            "positive": {k: v.squeeze(0) for k, v in positive.items()},
            "negative": {k: v.squeeze(0) for k, v in negative.items()},
            "weight": torch.tensor(sample_weight, dtype=torch.float32)
        }

    def _augment_text(self, text):
        words = text.split()
        if len(words) > 5:
            augmentation_type = random.random()

            if augmentation_type < 0.3:
                num_to_remove = max(1, len(words) // 15)
                indices_to_remove = random.sample(range(len(words)), min(num_to_remove, len(words) - 3))
                words = [word for i, word in enumerate(words) if i not in indices_to_remove]
            elif augmentation_type < 0.6:
                if len(words) > 3:
                    num_swaps = min(2, len(words) // 8)
                    for _ in range(num_swaps):
                        i, j = random.sample(range(len(words)), 2)
                        words[i], words[j] = words[j], words[i]
            else:
                synonyms = {"good": "great", "bad": "poor", "big": "large", "small": "tiny", "fast": "quick",
                            "slow": "sluggish"}
                words = [synonyms.get(word.lower(), word) if random.random() < 0.1 else word for word in words]

        return " ".join(words)

    def embed_texts(self, texts, model, tokenizer, device, batch_size=16):
        model.eval()
        embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
                batch = texts[i:i + batch_size]
                encoded = tokenizer(batch, padding=True, truncation=True, max_length=self.max_length,
                                    return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings.append(outputs.cpu())

        return torch.cat(embeddings, dim=0)

    def precompute_hard_negatives(self, model, tokenizer, device):
        all_anchor_texts = [sample["question"] for sample in self.samples]
        anchor_embeddings = self.embed_texts(all_anchor_texts, model, tokenizer, device)
        negative_pool_embeddings = self.embed_texts(self.negative_pool, model, tokenizer, device)

        sim_matrix = torch.matmul(anchor_embeddings, negative_pool_embeddings.T)

        sorted_indices = sim_matrix.argsort(dim=1, descending=True)

        hard_neg_indices = sorted_indices[:, 0]
        semi_hard_start = max(1, sim_matrix.size(1) // 4)
        semi_hard_end = min(sim_matrix.size(1), sim_matrix.size(1) // 2)

        semi_hard_range = torch.randint(semi_hard_start, semi_hard_end, (len(all_anchor_texts),))
        semi_hard_indices = sorted_indices[torch.arange(len(all_anchor_texts)), semi_hard_range]

        self.hard_negative_texts = [self.negative_pool[i] for i in hard_neg_indices]
        self.semi_hard_negative_texts = [self.negative_pool[i] for i in semi_hard_indices]

        positive_texts = [sample["positive"] for sample in self.samples]
        positive_embeddings = self.embed_texts(positive_texts, model, tokenizer, device)

        pos_similarities = torch.sum(anchor_embeddings * positive_embeddings, dim=1)
        hard_neg_similarities = sim_matrix[torch.arange(len(all_anchor_texts)), hard_neg_indices]

        margins = pos_similarities - hard_neg_similarities
        difficulty = torch.sigmoid(margins * 3)

        self.difficulty_weights = difficulty.tolist()

    def precompute_hard_positives(self, model, tokenizer, device):
        anchor_texts = [sample["question"] for sample in self.samples]
        all_texts = list(set(self.negative_pool + anchor_texts))
        anchor_embs = self.embed_texts(anchor_texts, model, tokenizer, device)
        all_embs = self.embed_texts(all_texts, model, tokenizer, device)

        sim_matrix = torch.matmul(anchor_embs, all_embs.T)
        top_indices = sim_matrix.topk(3, dim=1).indices

        for idx, sample in enumerate(self.samples):
            hard_pos = None
            for i in top_indices[idx]:
                candidate = all_texts[i]
                if candidate != sample["positive"] and candidate != sample["question"]:
                    hard_pos = candidate
                    break
            if hard_pos:
                sample["hard_positive"] = hard_pos
