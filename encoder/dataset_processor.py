import os
import json
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class DatasetProcessor(Dataset):
    def __init__(self, tokenizer, split, max_length, dataset, hard_negatives=True, hard_negative_ratio=0.4, semi_hard_ratio=0.3, random_neg_ratio=0.15):
        assert split in {"train", "test"}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_path = os.path.join("datasets", dataset)
        self.hard_negatives = hard_negatives
        self.hard_negative_ratio = hard_negative_ratio
        self.semi_hard_ratio = semi_hard_ratio
        self.random_neg_ratio = random_neg_ratio

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            all_samples = json.load(f)

        split_index = int(0.8 * len(all_samples))
        self.samples = all_samples[:split_index] if split == "train" else all_samples[split_index:]

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
        positive_text = item.get("hard_positive", item["positive"])

        rand = random.random()
        if self.hard_negatives and self.hard_negative_texts and rand < self.hard_negative_ratio:
            negative_text = self.hard_negative_texts[idx]
        elif self.hard_negatives and self.semi_hard_negative_texts and rand < self.hard_negative_ratio + self.semi_hard_ratio:
            negative_text = self.semi_hard_negative_texts[idx]
        elif self.hard_negatives and rand < self.hard_negative_ratio + self.semi_hard_ratio + self.random_neg_ratio:
            negative_text = random.choice(self.negative_pool)
        else:
            negative_text = item["negative"]

        if random.random() < 0.05:
            anchor_text = self._augment_text(anchor_text)
        if random.random() < 0.05:
            positive_text = self._augment_text(positive_text)
        if random.random() < 0.03:
            negative_text = self._augment_text(negative_text)

        anchor = self.tokenizer(anchor_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        positive = self.tokenizer(positive_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        negative = self.tokenizer(negative_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        sample_weight = 1.0
        if self.difficulty_weights:
            sample_weight = self.difficulty_weights[idx]

        return {
            "anchor": {k: v.squeeze(0) for k, v in anchor.items()},
            "positive": {k: v.squeeze(0) for k, v in positive.items()},
            "negative": {k: v.squeeze(0) for k, v in negative.items()},
            "weight": torch.tensor(sample_weight, dtype=torch.float32)
        }

    def _augment_text(self, text):
        words = text.split()
        if len(words) <= 5:
            return text
        t = random.random()
        if t < 0.3:
            num = max(1, len(words) // 15)
            indices = random.sample(range(len(words)), min(num, len(words) - 3))
            words = [w for i, w in enumerate(words) if i not in indices]
        elif t < 0.6:
            for _ in range(min(2, len(words) // 8)):
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
        else:
            synonyms = {
                "tribunal": "instância",
                "recurso": "apelação",
                "decisão": "despacho",
                "sentença": "acórdão",
                "parte": "interveniente",
                "lei": "norma",
                "bom": "favorável",
                "mau": "desfavorável"
            }
            words = [synonyms.get(w.lower(), w) if random.random() < 0.1 else w for w in words]
        return " ".join(words)

    def embed_texts(self, texts, model, tokenizer, device, batch_size=16):
        model.eval()
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
                batch = texts[i:i + batch_size]
                encoded = tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(device)
                emb = model(encoded['input_ids'], encoded['attention_mask'])
                embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0)

    def precompute_hard_negatives(self, model, tokenizer, device):
        cache_path = self.dataset_path.replace(".json", "_hardcache.pt")
        if os.path.exists(cache_path):
            cache = torch.load(cache_path)
            self.hard_negative_texts = cache['hard']
            self.semi_hard_negative_texts = cache['semi']
            self.difficulty_weights = cache['weights']
            return

        anchors = [s["question"] for s in self.samples]
        positives = [s["positive"] for s in self.samples]
        anchor_embs = self.embed_texts(anchors, model, tokenizer, device)
        positive_embs = self.embed_texts(positives, model, tokenizer, device)
        negative_pool_embs = self.embed_texts(self.negative_pool, model, tokenizer, device)

        sim_matrix = torch.matmul(anchor_embs, negative_pool_embs.T)
        hard_threshold = torch.quantile(sim_matrix, 0.95).item()
        semi_hard_low = torch.quantile(sim_matrix, 0.70).item()
        semi_hard_high = torch.quantile(sim_matrix, 0.85).item()

        hard_neg_indices = sim_matrix.argmax(dim=1)
        semi_hard_indices = []

        for i in range(sim_matrix.size(0)):
            sims = sim_matrix[i]
            mask = (sims > semi_hard_low) & (sims < semi_hard_high)
            candidates = torch.nonzero(mask).squeeze()
            if candidates.numel() > 0:
                chosen = candidates[random.randint(0, len(candidates) - 1)]
            else:
                chosen = random.randint(0, len(self.negative_pool) - 1)
            semi_hard_indices.append(chosen.item())

        hard_neg_indices = hard_neg_indices.tolist()
        semi_hard_indices = semi_hard_indices

        self.hard_negative_texts = [self.negative_pool[i] for i in hard_neg_indices]
        self.semi_hard_negative_texts = [self.negative_pool[i] for i in semi_hard_indices]

        pos_sims = torch.sum(anchor_embs * positive_embs, dim=1)
        neg_sims = sim_matrix[torch.arange(len(self.samples)), torch.tensor(hard_neg_indices)]
        margins = pos_sims - neg_sims
        scale = 1 / (margins.std().item() + 1e-6)
        self.difficulty_weights = torch.sigmoid(margins * scale).tolist()

        torch.save({
            'hard': self.hard_negative_texts,
            'semi': self.semi_hard_negative_texts,
            'weights': self.difficulty_weights
        }, cache_path)

    def precompute_hard_positives(self, model, tokenizer, device):
        anchors = [s["question"] for s in self.samples]
        all_texts = list(set(self.negative_pool + anchors))
        anchor_embs = self.embed_texts(anchors, model, tokenizer, device)
        all_embs = self.embed_texts(all_texts, model, tokenizer, device)

        sim_matrix = torch.matmul(anchor_embs, all_embs.T)
        for i, s in enumerate(self.samples):
            sorted_indices = torch.argsort(sim_matrix[i], descending=True)
            for j in sorted_indices:
                candidate = all_texts[j]
                if candidate != s["positive"] and candidate != s["question"]:
                    sim_score = sim_matrix[i][j].item()
                    if sim_score > 0.7:
                        s["hard_positive"] = candidate
                        break
