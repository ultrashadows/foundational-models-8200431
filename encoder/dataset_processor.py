import random
import logging
import json
import os
from typing import Dict, List, Optional
from torch.utils.data import Dataset
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


class DatasetProcessor(Dataset):
    def __init__(self, tokenizer, max_length=512, split="train", train_ratio=0.8, max_samples=None,
                 seed=42, json_path="dataset.json"):
        logger.info(f"Initializing DatasetProcessor with split='{split}', max_samples={max_samples}")
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = self._process_dataset(json_path, split, train_ratio, max_samples, seed)
        logger.info(f"Dataset ready with {len(self.samples)} samples")

    def _process_dataset(self, json_path: str, split: str, train_ratio: float,
                         max_samples: Optional[int], seed: int) -> List[Dict[str, str]]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        def extract_id(value):
            if isinstance(value, dict) and '$oid' in value:
                return value['$oid']
            return str(value)

        doc_map = {extract_id(doc['_id']): doc for doc in data}
        all_samples = []

        logger.info(f"Processing {len(data)} documents...")
        processed_count = 0

        for doc in data:
            if 'results' in doc and isinstance(doc['results'], list) and doc.get('wrong'):
                correct_doc_text = doc.get('Texto Integral', '')
                if not correct_doc_text:
                    continue

                wrong_id = extract_id(doc['wrong'])
                wrong_doc = doc_map.get(wrong_id)
                if not wrong_doc:
                    continue

                wrong_doc_text = wrong_doc.get('Texto Integral', '')
                if not wrong_doc_text:
                    continue

                max_text_length = 3000
                if len(correct_doc_text) > max_text_length:
                    correct_doc_text = correct_doc_text[:max_text_length] + "..."
                if len(wrong_doc_text) > max_text_length:
                    wrong_doc_text = wrong_doc_text[:max_text_length] + "..."

                for qa in doc['results']:
                    if 'pergunta' in qa and qa['pergunta'].strip():
                        all_samples.append({
                            'anchor': qa['pergunta'].strip(),
                            'positive': correct_doc_text,
                            'negative': wrong_doc_text
                        })

            processed_count += 1
            if processed_count % 1000 == 0:
                logger.info(f"Processed {processed_count}/{len(data)} documents")

        logger.info(f"Total samples before split/filtering: {len(all_samples)}")

        random.seed(seed)
        random.shuffle(all_samples)

        if max_samples is not None and len(all_samples) > max_samples:
            all_samples = all_samples[:max_samples]
            logger.info(f"Trimmed to max_samples={max_samples}")

        train_size = int(len(all_samples) * train_ratio)

        if split == "train":
            samples = all_samples[:train_size]
        elif split == "test":
            samples = all_samples[train_size:]
        else:
            samples = all_samples

        logger.info(f"Created '{split}' dataset with {len(samples)} samples.")
        return samples

    def _tokenize_sample(self, sample: Dict[str, str]) -> Dict[str, torch.Tensor]:
        anchor_encoding = self.tokenizer(
            sample['anchor'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True
        )

        positive_encoding = self.tokenizer(
            sample['positive'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True
        )

        negative_encoding = self.tokenizer(
            sample['negative'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True
        )

        return {
            'anchor': {k: v.squeeze(0) for k, v in anchor_encoding.items()},
            'positive': {k: v.squeeze(0) for k, v in positive_encoding.items()},
            'negative': {k: v.squeeze(0) for k, v in negative_encoding.items()}
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._tokenize_sample(self.samples[idx])

    def get_sample_text(self, idx: int) -> Dict[str, str]:
        return self.samples[idx]

    def get_stats(self) -> Dict[str, any]:
        if not self.samples:
            return {}

        anchor_lengths = [len(sample['anchor'].split()) for sample in self.samples]
        positive_lengths = [len(sample['positive'].split()) for sample in self.samples]
        negative_lengths = [len(sample['negative'].split()) for sample in self.samples]

        return {
            'total_samples': len(self.samples),
            'avg_anchor_length': sum(anchor_lengths) / len(anchor_lengths),
            'avg_positive_length': sum(positive_lengths) / len(positive_lengths),
            'avg_negative_length': sum(negative_lengths) / len(negative_lengths),
            'max_anchor_length': max(anchor_lengths),
            'max_positive_length': max(positive_lengths),
            'max_negative_length': max(negative_lengths)
        }
