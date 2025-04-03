from torch.utils.data import Dataset


class DatasetProcessor(Dataset):
    def __init__(self, collection, tokenizer, max_length=1024):
        self.collection = collection
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = list(self.collection.find({}))
        self.samples = []

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
                        self.samples.append({
                            'anchor': qa['pergunta'],
                            'positive': correct_doc_text,
                            'negative': wrong_doc_text
                        })

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
