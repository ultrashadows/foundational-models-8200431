import random
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DatasetProcessor(Dataset):
    def __init__(self, collection, tokenizer, max_length=1024, split="train", train_ratio=0.8, max_samples=None,
                 seed=42):
        logger.info(f"Initializing DatasetProcessor with split='{split}', max_samples={max_samples}")
        self.collection = collection
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = list(self.collection.find({}))
        all_samples = []

        for doc in self.data:
            if 'results' in doc and isinstance(doc['results'], list) and doc.get('wrong'):
                correct_doc_text = doc.get('Texto Integral', '')

                if not correct_doc_text:
                    continue

                wrong_doc = self.collection.find_one({'_id': doc['wrong']})

                if not wrong_doc:
                    continue

                wrong_doc_text = wrong_doc.get('Texto Integral', '')

                if not wrong_doc_text:
                    continue

                for qa in doc['results']:
                    if 'pergunta' in qa:
                        all_samples.append({
                            'anchor': qa['pergunta'],
                            'positive': correct_doc_text,
                            'negative': wrong_doc_text
                        })

        logger.info(f"Total samples before split/filtering: {len(all_samples)}")
        random.seed(seed)
        random.shuffle(all_samples)

        if max_samples is not None and len(all_samples) > max_samples:
            all_samples = all_samples[:max_samples]
            logger.info(f"Trimmed to max_samples={max_samples}")

        train_size = int(len(all_samples) * train_ratio)

        if split == "train":
            self.samples = all_samples[:train_size]
        elif split == "test":
            self.samples = all_samples[train_size:]
        else:
            self.samples = all_samples

        logger.info(f"Created '{split}' dataset with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        anchor_encoding = self.tokenizer(
            sample['anchor'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        positive_encoding = self.tokenizer(
            sample['positive'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        negative_encoding = self.tokenizer(
            sample['negative'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'anchor': {k: v.squeeze(0) for k, v in anchor_encoding.items()},
            'positive': {k: v.squeeze(0) for k, v in positive_encoding.items()},
            'negative': {k: v.squeeze(0) for k, v in negative_encoding.items()}
        }
