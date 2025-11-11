import os
import json
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class DatasetProcessor(Dataset):
    def __init__(self, tokenizer, split, max_length, dataset, hard_negatives=False,
                 hard_negative_ratio=0.35, semi_hard_ratio=0.45, easy_ratio=0.10, random_neg_ratio=0.10):
        assert split in {"train", "test"}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_path = os.path.join("datasets", dataset)
        self.hard_negatives = hard_negatives

        self.hard_negative_ratio = hard_negative_ratio
        self.semi_hard_ratio = semi_hard_ratio
        self.easy_ratio = easy_ratio
        self.random_neg_ratio = random_neg_ratio

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            all_samples = json.load(f)

        split_index = int(0.9 * len(all_samples))
        self.samples = all_samples[:split_index] if split == "train" else all_samples[split_index:]

        self.negative_pool = []
        for sample in self.samples:
            if "negative" in sample:
                self.negative_pool.append(sample["negative"])
            if "positive" in sample:
                self.negative_pool.append(sample["positive"])

        self.hard_negative_texts = None
        self.semi_hard_negative_texts = None
        self.easy_negative_texts = None
        self.difficulty_weights = None

        if split == "train":
            random.shuffle(self.samples)
            print(f"Loaded {len(self.samples):,} training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        anchor_text = item["question"]
        positive_text = item.get("hard_positive", item["positive"])

        rand = random.random()
        cumulative_semi = self.semi_hard_ratio
        cumulative_hard = cumulative_semi + self.hard_negative_ratio
        cumulative_easy = cumulative_hard + self.easy_ratio

        if self.hard_negatives and self.semi_hard_negative_texts and rand < cumulative_semi:
            negative_text = self.semi_hard_negative_texts[idx]
        elif self.hard_negatives and self.hard_negative_texts and rand < cumulative_hard:
            negative_text = self.hard_negative_texts[idx]
        elif self.hard_negatives and self.easy_negative_texts and rand < cumulative_easy:
            negative_text = self.easy_negative_texts[idx]
        elif rand < cumulative_easy + self.random_neg_ratio:
            negative_text = random.choice(self.negative_pool)
        else:
            negative_text = item["negative"]

        if random.random() < 0.03:
            anchor_text = self._augment_text(anchor_text)
        if random.random() < 0.03:
            positive_text = self._augment_text(positive_text)
        if random.random() < 0.02:
            negative_text = self._augment_text(negative_text)

        anchor = self.tokenizer(anchor_text, padding="max_length", truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        positive = self.tokenizer(positive_text, padding="max_length", truncation=True,
                                  max_length=self.max_length, return_tensors="pt")
        negative = self.tokenizer(negative_text, padding="max_length", truncation=True,
                                  max_length=self.max_length, return_tensors="pt")

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
        if t < 0.4:
            num = max(1, len(words) // 20)
            indices = random.sample(range(len(words)), min(num, len(words) - 3))
            words = [w for i, w in enumerate(words) if i not in indices]
        elif t < 0.7:
            for _ in range(min(1, len(words) // 10)):
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
                "processo": "procedimento",
                "juiz": "magistrado",
                "arguido": "réu"
            }
            words = [synonyms.get(w.lower(), w) if random.random() < 0.08 else w for w in words]

        return " ".join(words)

    def embed_texts(self, texts, model, tokenizer, device, batch_size=16):
        model.eval()
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
                batch = texts[i:i + batch_size]
                encoded = tokenizer(batch, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt").to(device)
                emb = model(encoded['input_ids'], encoded['attention_mask'])
                embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0)

    def precompute_hard_negatives(self, model, tokenizer, device):
        cache_path = self.dataset_path.replace(".json", "_hardcache.pt")
        if os.path.exists(cache_path):
            cache = torch.load(cache_path)
            self.hard_negative_texts = cache['hard']
            self.semi_hard_negative_texts = cache['semi']
            self.easy_negative_texts = cache.get('easy', cache['semi'])
            self.difficulty_weights = cache['weights']
            print(f"Loaded hard negatives from cache")
            return

        print("Computing hard negatives for 100K dataset...")
        anchors = [s["question"] for s in self.samples]
        positives = [s["positive"] for s in self.samples]

        anchor_embs = self.embed_texts(anchors, model, tokenizer, device, batch_size=32)
        positive_embs = self.embed_texts(positives, model, tokenizer, device, batch_size=32)
        negative_pool_embs = self.embed_texts(self.negative_pool, model, tokenizer, device, batch_size=32)

        sim_matrix = torch.matmul(anchor_embs, negative_pool_embs.T)
        sorted_sims, sorted_indices = torch.sort(sim_matrix, dim=1, descending=True)

        hard_neg_indices = []
        semi_hard_neg_indices = []
        easy_neg_indices = []

        for i in range(sim_matrix.size(0)):
            hard_start = int(sim_matrix.size(1) * 0.05)
            hard_end = int(sim_matrix.size(1) * 0.15)
            if hard_start < hard_end < sorted_indices.size(1):
                hard_idx = random.randint(hard_start, min(hard_end - 1, sorted_indices.size(1) - 1))
                hard_neg_indices.append(sorted_indices[i, hard_idx].item())
            else:
                hard_neg_indices.append(sorted_indices[i, min(hard_start, sorted_indices.size(1) - 1)].item())

            semi_start = int(sim_matrix.size(1) * 0.15)
            semi_end = int(sim_matrix.size(1) * 0.35)
            if semi_start < semi_end < sorted_indices.size(1):
                semi_idx = random.randint(semi_start, min(semi_end - 1, sorted_indices.size(1) - 1))
                semi_hard_neg_indices.append(sorted_indices[i, semi_idx].item())
            else:
                semi_hard_neg_indices.append(sorted_indices[i, min(semi_start, sorted_indices.size(1) - 1)].item())

            easy_start = int(sim_matrix.size(1) * 0.50)
            easy_end = int(sim_matrix.size(1) * 0.70)
            if easy_start < easy_end < sorted_indices.size(1):
                easy_idx = random.randint(easy_start, min(easy_end - 1, sorted_indices.size(1) - 1))
                easy_neg_indices.append(sorted_indices[i, easy_idx].item())
            else:
                easy_neg_indices.append(sorted_indices[i, min(easy_start, sorted_indices.size(1) - 1)].item())

        self.hard_negative_texts = [self.negative_pool[i] for i in hard_neg_indices]
        self.semi_hard_negative_texts = [self.negative_pool[i] for i in semi_hard_neg_indices]
        self.easy_negative_texts = [self.negative_pool[i] for i in easy_neg_indices]

        pos_sims = torch.sum(anchor_embs * positive_embs, dim=1)
        neg_sims = sim_matrix[torch.arange(len(self.samples)), torch.tensor(semi_hard_neg_indices)]
        margins = pos_sims - neg_sims

        margins = torch.clamp(margins, min=-2.0, max=2.0)
        scale = 1 / (margins.std().item() + 1e-6)
        self.difficulty_weights = torch.sigmoid(margins * scale).tolist()

        torch.save({
            'hard': self.hard_negative_texts,
            'semi': self.semi_hard_negative_texts,
            'easy': self.easy_negative_texts,
            'weights': self.difficulty_weights
        }, cache_path)

        print(
            f"Hard negatives - Mean sim: {sim_matrix[torch.arange(len(self.samples)), torch.tensor(hard_neg_indices)].mean():.4f}")
        print(
            f"Semi-hard negatives - Mean sim: {sim_matrix[torch.arange(len(self.samples)), torch.tensor(semi_hard_neg_indices)].mean():.4f}")
        print(
            f"Easy negatives - Mean sim: {sim_matrix[torch.arange(len(self.samples)), torch.tensor(easy_neg_indices)].mean():.4f}")
        print(f"Positive similarities - Mean: {pos_sims.mean():.4f}")
        print(f"Margins - Mean: {margins.mean():.4f}, Std: {margins.std():.4f}")