import random
import logging
import json
import os
import re
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from threading import Lock
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


class DatasetProcessor(Dataset):
    def __init__(self, tokenizer=None, max_length=512, split="train", train_ratio=0.8, max_samples=None,
                 seed=42, json_path="dataset.json", chunk_overlap=50, use_chunking_threshold=400,
                 max_chunks_per_sample=2, min_chunk_tokens=100, preserve_structure=True,
                 cache_dir="cache", enable_cache=True, num_workers=4, batch_size=100):
        logger.info(f"Initializing Enhanced HybridDatasetProcessor with split='{split}', max_samples={max_samples}")
        self.tokenizer = AutoTokenizer.from_pretrained("./models/albertina-1b5") if tokenizer is None else tokenizer
        self.max_length = max_length
        self.chunk_overlap = chunk_overlap
        self.use_chunking_threshold = use_chunking_threshold
        self.max_chunks_per_sample = max_chunks_per_sample
        self.min_chunk_tokens = min_chunk_tokens
        self.preserve_structure = preserve_structure
        self.effective_doc_length = max_length - 20
        self.cache_dir = cache_dir
        self.enable_cache = enable_cache
        self.json_path = json_path
        self.split = split
        self.train_ratio = train_ratio
        self.max_samples = max_samples
        self.seed = seed
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.cache_lock = Lock()
        self.progress_lock = Lock()

        self.progress_stats = {
            'processed': 0,
            'cache_hits': 0,
            'new_processed': 0,
            'errors': 0
        }

        self.legal_patterns = self._init_legal_patterns()

        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.samples = self._process_dataset(json_path, split, train_ratio, max_samples, seed)
        logger.info(f"Dataset ready with {len(self.samples)} samples")

    @classmethod
    def preprocess_only(cls, json_path: str, cache_dir: str = "cache", max_length: int = 512,
                        chunk_overlap: int = 50, use_chunking_threshold: int = 400,
                        max_chunks_per_sample: int = 2, min_chunk_tokens: int = 100,
                        preserve_structure: bool = True, force_refresh: bool = False,
                        num_workers: int = 4, batch_size: int = 100):

        logger.info(f"Starting preprocessing-only mode with {num_workers} workers")

        processor = cls(
            tokenizer=AutoTokenizer.from_pretrained("./models/albertina-1b5"),
            max_length=max_length,
            split="all",
            train_ratio=0.8,
            max_samples=None,
            seed=42,
            json_path=json_path,
            chunk_overlap=chunk_overlap,
            use_chunking_threshold=use_chunking_threshold,
            max_chunks_per_sample=max_chunks_per_sample,
            min_chunk_tokens=min_chunk_tokens,
            preserve_structure=preserve_structure,
            cache_dir=cache_dir,
            enable_cache=True,
            num_workers=num_workers,
            batch_size=batch_size
        )

        if force_refresh:
            processor.clear_cache()
            logger.info("Cache cleared, starting fresh preprocessing")

        processor._preprocess_all_documents()

        stats = processor.get_cache_stats()
        logger.info(f"Preprocessing completed. Cache stats: {stats}")

        return processor

    def _create_worker_tokenizer(self):
        return AutoTokenizer.from_pretrained("./models/albertina-1b5")

    def _process_document_batch(self, doc_batch: List[Tuple[Dict, Dict]],
                                processing_params: Dict) -> List[Optional[Dict]]:
        local_tokenizer = self._create_worker_tokenizer()
        original_tokenizer = self.tokenizer
        self.tokenizer = local_tokenizer

        results = []
        local_stats = {'processed': 0, 'cache_hits': 0, 'new_processed': 0, 'errors': 0}

        try:
            for doc, wrong_doc in doc_batch:
                try:
                    doc_id = self._extract_id(doc['_id'])
                    wrong_id = self._extract_id(wrong_doc['_id'])

                    cache_key = self._get_cache_key(f"{doc_id}_{wrong_id}", processing_params)

                    with self.cache_lock:
                        cached_result = self._load_from_cache(cache_key)

                    if cached_result is not None:
                        local_stats['cache_hits'] += 1
                        results.append(cached_result)
                    else:
                        processed_result = self._process_document_without_cache(doc, wrong_doc)
                        if processed_result is not None:
                            with self.cache_lock:
                                self._save_to_cache(cache_key, processed_result)
                            local_stats['new_processed'] += 1
                            results.append(processed_result)
                        else:
                            results.append(None)

                    local_stats['processed'] += 1

                except Exception as e:
                    logger.error(f"Error processing document {doc.get('_id', 'unknown')}: {e}")
                    local_stats['errors'] += 1
                    results.append(None)

        finally:
            self.tokenizer = original_tokenizer

        with self.progress_lock:
            for key, value in local_stats.items():
                self.progress_stats[key] += value

        return results

    def _process_document_without_cache(self, doc: Dict, wrong_doc: Dict) -> Optional[Dict]:
        correct_doc_text = doc.get('Texto Integral', '')
        wrong_doc_text = wrong_doc.get('Texto Integral', '')

        if not correct_doc_text or not wrong_doc_text:
            return None

        correct_metadata = self._extract_document_metadata(doc)
        wrong_metadata = self._extract_document_metadata(wrong_doc)

        should_chunk_positive = self._should_chunk(correct_doc_text)
        should_chunk_negative = self._should_chunk(wrong_doc_text)

        if should_chunk_positive or should_chunk_negative:
            correct_chunks = self._tokenize_and_chunk(correct_doc_text,
                                                      correct_metadata) if should_chunk_positive else [correct_doc_text]
            wrong_chunks = self._tokenize_and_chunk(wrong_doc_text, wrong_metadata) if should_chunk_negative else [
                wrong_doc_text]

            result = {
                'correct_chunks': correct_chunks,
                'wrong_chunks': wrong_chunks,
                'is_chunked': True,
                'correct_doc_text': correct_doc_text,
                'wrong_doc_text': wrong_doc_text
            }
        else:
            result = {
                'correct_chunks': [correct_doc_text],
                'wrong_chunks': [wrong_doc_text],
                'is_chunked': False,
                'correct_doc_text': correct_doc_text,
                'wrong_doc_text': wrong_doc_text
            }

        return result

    def _create_document_batches(self, doc_pairs: List[Tuple[Dict, Dict]]) -> List[List[Tuple[Dict, Dict]]]:
        batches = []
        for i in range(0, len(doc_pairs), self.batch_size):
            batch = doc_pairs[i:i + self.batch_size]
            batches.append(batch)
        return batches

    def _preprocess_all_documents(self):
        logger.info(f"Starting complete document preprocessing with {self.num_workers} workers")

        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        doc_map = {self._extract_id(doc['_id']): doc for doc in data}
        processing_params = self._get_processing_params()

        doc_pairs = []
        for doc in data:
            if 'results' in doc and isinstance(doc['results'], list) and doc.get('wrong'):
                wrong_id = doc['wrong']['$oid']
                wrong_doc = doc_map.get(wrong_id)
                if wrong_doc:
                    doc_pairs.append((doc, wrong_doc))

        total_docs = len(doc_pairs)
        logger.info(f"Found {total_docs} document pairs to process")

        batches = self._create_document_batches(doc_pairs)
        logger.info(f"Created {len(batches)} batches with batch size {self.batch_size}")

        with self.progress_lock:
            self.progress_stats = {'processed': 0, 'cache_hits': 0, 'new_processed': 0, 'errors': 0}

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_document_batch, batch, processing_params): i
                for i, batch in enumerate(batches)
            }

            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()

                    if (batch_idx + 1) % 10 == 0:
                        with self.progress_lock:
                            current_stats = self.progress_stats.copy()

                        logger.info(f"Processed {batch_idx + 1}/{len(batches)} batches "
                                    f"(Docs: {current_stats['processed']}/{total_docs}, "
                                    f"Cache hits: {current_stats['cache_hits']}, "
                                    f"New: {current_stats['new_processed']}, "
                                    f"Errors: {current_stats['errors']})")

                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")

        with self.progress_lock:
            final_stats = self.progress_stats.copy()

        logger.info(f"Preprocessing complete: {final_stats['processed']} documents processed, "
                    f"{final_stats['cache_hits']} cache hits, {final_stats['new_processed']} newly processed, "
                    f"{final_stats['errors']} errors")

    def _process_dataset_parallel(self, data: List[Dict], doc_map: Dict[str, Dict],
                                  processing_params: Dict) -> List[Dict[str, str]]:
        logger.info(f"Processing dataset with {self.num_workers} workers...")

        doc_pairs = []
        for doc in data:
            if 'results' in doc and isinstance(doc['results'], list) and doc.get('wrong'):
                wrong_id = doc['wrong']['$oid']
                wrong_doc = doc_map.get(wrong_id)
                if wrong_doc:
                    doc_pairs.append((doc, wrong_doc))

        batches = self._create_document_batches(doc_pairs)
        all_samples = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_sample_batch, batch, processing_params): i
                for i, batch in enumerate(batches)
            }

            for future in as_completed(future_to_batch):
                try:
                    batch_samples = future.result()
                    all_samples.extend(batch_samples)
                except Exception as e:
                    logger.error(f"Error processing sample batch: {e}")

        return all_samples

    def _process_sample_batch(self, doc_batch: List[Tuple[Dict, Dict]],
                              processing_params: Dict) -> List[Dict[str, str]]:
        local_tokenizer = self._create_worker_tokenizer()
        original_tokenizer = self.tokenizer
        self.tokenizer = local_tokenizer

        batch_samples = []

        try:
            for doc, wrong_doc in doc_batch:
                doc_id = self._extract_id(doc['_id'])
                wrong_id = self._extract_id(wrong_doc['_id'])

                cache_key = self._get_cache_key(f"{doc_id}_{wrong_id}", processing_params)

                with self.cache_lock:
                    cached_result = self._load_from_cache(cache_key)

                if cached_result is None:
                    cached_result = self._process_document_without_cache(doc, wrong_doc)
                    if cached_result is not None:
                        with self.cache_lock:
                            self._save_to_cache(cache_key, cached_result)

                if cached_result is None:
                    continue

                for qa in doc.get('results', []):
                    if 'pergunta' in qa and qa['pergunta'].strip():
                        question = qa['pergunta'].strip()

                        if cached_result['is_chunked']:
                            relevant_positive = self._find_relevant_chunks(question, cached_result['correct_chunks'])
                            relevant_negative = self._find_relevant_chunks(question, cached_result['wrong_chunks'])

                            combined_positive = self._combine_chunks(relevant_positive)
                            combined_negative = self._combine_chunks(relevant_negative)

                            batch_samples.append({
                                'anchor': question,
                                'positive': combined_positive,
                                'negative': combined_negative,
                                'is_chunked': True,
                                'num_positive_chunks': len(relevant_positive),
                                'num_negative_chunks': len(relevant_negative)
                            })
                        else:
                            batch_samples.append({
                                'anchor': question,
                                'positive': cached_result['correct_doc_text'],
                                'negative': cached_result['wrong_doc_text'],
                                'is_chunked': False,
                                'num_positive_chunks': 1,
                                'num_negative_chunks': 1
                            })

        finally:
            self.tokenizer = original_tokenizer

        return batch_samples

    def _init_legal_patterns(self) -> Dict[str, re.Pattern]:
        return {
            'article': re.compile(r'(?:^|\n)\s*(?:Art\.|Artigo)\s*\d+', re.IGNORECASE | re.MULTILINE),
            'section': re.compile(r'(?:^|\n)\s*(?:Seção|Secção)\s*[IVX\d]+', re.IGNORECASE | re.MULTILINE),
            'paragraph': re.compile(r'(?:^|\n)\s*§\s*\d+', re.MULTILINE),
            'item': re.compile(r'(?:^|\n)\s*[IVX]+\s*[-–—]\s*', re.MULTILINE),
            'chapter': re.compile(r'(?:^|\n)\s*(?:Capítulo|Cap\.)\s*[IVX\d]+', re.IGNORECASE | re.MULTILINE),
            'title': re.compile(r'(?:^|\n)\s*(?:Título|Tít\.)\s*[IVX\d]+', re.IGNORECASE | re.MULTILINE)
        }

    def _get_cache_key(self, doc_id: str, params: Dict) -> str:
        param_str = json.dumps(params, sort_keys=True)
        combined = f"{doc_id}_{param_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        if not self.enable_cache:
            return None

        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
                return None
        return None

    def _save_to_cache(self, cache_key: str, data: Dict) -> None:
        if not self.enable_cache:
            return

        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

    def _get_processing_params(self) -> Dict:
        tokenizer_name = getattr(self.tokenizer, 'name_or_path', 'unknown')
        if hasattr(self.tokenizer, '__class__'):
            tokenizer_name = f"{tokenizer_name}_{self.tokenizer.__class__.__name__}"

        return {
            'max_length': self.max_length,
            'chunk_overlap': self.chunk_overlap,
            'use_chunking_threshold': self.use_chunking_threshold,
            'max_chunks_per_sample': self.max_chunks_per_sample,
            'min_chunk_tokens': self.min_chunk_tokens,
            'preserve_structure': self.preserve_structure,
            'effective_doc_length': self.effective_doc_length,
            'tokenizer_name': tokenizer_name
        }

    def _should_chunk(self, text: str) -> bool:
        token_count = len(self.tokenizer.tokenize(text))
        return token_count > self.use_chunking_threshold

    def _extract_document_metadata(self, doc: Dict) -> str:
        metadata_parts = []

        if 'numero' in doc:
            metadata_parts.append(f"Documento nº {doc['numero']}")
        if 'ementa' in doc:
            metadata_parts.append(f"Ementa: {doc['ementa'][:100]}...")
        if 'tribunal' in doc:
            metadata_parts.append(f"Tribunal: {doc['tribunal']}")
        if 'data' in doc:
            metadata_parts.append(f"Data: {doc['data']}")

        return " | ".join(metadata_parts) if metadata_parts else ""

    def _find_structural_boundaries(self, text: str) -> List[int]:
        boundaries = [0]

        for pattern_name, pattern in self.legal_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                boundaries.append(match.start())

        sentence_pattern = re.compile(r'[.!?]\s+')
        matches = sentence_pattern.finditer(text)
        for match in matches:
            boundaries.append(match.end())

        return sorted(list(set(boundaries)))

    def _tokenize_and_chunk(self, text: str, metadata: str = "") -> List[str]:
        tokens = self.tokenizer.tokenize(text)

        if len(tokens) <= self.effective_doc_length:
            return [text]

        chunks = []

        if self.preserve_structure:
            boundaries = self._find_structural_boundaries(text)
            chunks = self._chunk_by_structure(text, tokens, boundaries, metadata)
        else:
            chunks = self._chunk_by_tokens(text, tokens, metadata)

        return chunks if chunks else [text]

    def _chunk_by_structure(self, text: str, tokens: List[str], boundaries: List[int], metadata: str) -> List[str]:
        chunks = []
        current_token_start = 0

        for boundary in boundaries[1:] + [len(text)]:
            char_to_token = self._map_chars_to_tokens(text, tokens)
            boundary_token = char_to_token.get(boundary, len(tokens))

            chunk_tokens = tokens[current_token_start:boundary_token]

            if len(chunk_tokens) >= self.min_chunk_tokens:
                if len(chunk_tokens) > self.effective_doc_length:
                    sub_chunks = self._split_large_chunk(chunk_tokens, metadata)
                    chunks.extend(sub_chunks)
                else:
                    chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                    if metadata:
                        chunk_text = f"{metadata}\n\n{chunk_text}"
                    chunks.append(chunk_text)

                current_token_start = max(0, boundary_token - self.chunk_overlap)

            if boundary_token >= len(tokens):
                break

        return chunks

    def _chunk_by_tokens(self, text: str, tokens: List[str], metadata: str) -> List[str]:
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + self.effective_doc_length, len(tokens))
            chunk_tokens = tokens[start:end]

            if len(chunk_tokens) >= self.min_chunk_tokens:
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                if metadata:
                    chunk_text = f"{metadata}\n\n{chunk_text}"
                chunks.append(chunk_text)

            start = end - self.chunk_overlap

            if end >= len(tokens):
                break

        return chunks

    def _split_large_chunk(self, chunk_tokens: List[str], metadata: str) -> List[str]:
        sub_chunks = []
        start = 0

        while start < len(chunk_tokens):
            end = min(start + self.effective_doc_length, len(chunk_tokens))
            sub_chunk_tokens = chunk_tokens[start:end]

            if len(sub_chunk_tokens) >= self.min_chunk_tokens:
                sub_chunk_text = self.tokenizer.convert_tokens_to_string(sub_chunk_tokens)
                if metadata:
                    sub_chunk_text = f"{metadata}\n\n{sub_chunk_text}"
                sub_chunks.append(sub_chunk_text)

            start = end - self.chunk_overlap

            if end >= len(chunk_tokens):
                break

        return sub_chunks

    def _map_chars_to_tokens(self, text: str, tokens: List[str]) -> Dict[int, int]:
        char_to_token = {}
        current_pos = 0

        for i, token in enumerate(tokens):
            token_text = token.replace('▁', ' ').replace('##', '')

            while current_pos < len(text) and text[current_pos:current_pos + len(token_text)] != token_text:
                current_pos += 1

            if current_pos < len(text):
                char_to_token[current_pos] = i
                current_pos += len(token_text)

        return char_to_token

    def _find_relevant_chunks(self, question: str, chunks: List[str], max_chunks: int = None) -> List[str]:
        if max_chunks is None:
            max_chunks = self.max_chunks_per_sample

        if len(chunks) <= max_chunks:
            return chunks

        chunk_scores = []
        question_lower = question.lower()
        question_words = set(question_lower.split())

        legal_terms = {'lei', 'artigo', 'código', 'decreto', 'jurisprudência', 'tribunal', 'decisão',
                       'processo', 'recurso', 'apelação', 'embargo', 'habeas', 'mandado', 'ação'}

        for chunk in chunks:
            chunk_lower = chunk.lower()
            chunk_words = set(chunk_lower.split())

            word_overlap = len(question_words.intersection(chunk_words))
            legal_overlap = len(question_words.intersection(legal_terms))

            phrase_bonus = 0
            for word in question_words:
                if len(word) > 3 and word in chunk_lower:
                    phrase_bonus += 1

            context_score = self._calculate_context_score(question_lower, chunk_lower)

            total_score = word_overlap + (legal_overlap * 2) + phrase_bonus + context_score
            chunk_scores.append((total_score, chunk))

        chunk_scores.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in chunk_scores[:max_chunks]]

    def _calculate_context_score(self, question: str, chunk: str) -> float:
        question_words = question.split()
        chunk_words = chunk.split()

        context_score = 0
        for q_word in question_words:
            if q_word in chunk_words:
                positions = [i for i, word in enumerate(chunk_words) if word == q_word]
                for pos in positions:
                    nearby_words = chunk_words[max(0, pos - 3):pos + 4]
                    for other_q_word in question_words:
                        if other_q_word in nearby_words and other_q_word != q_word:
                            context_score += 0.5

        return context_score

    def _combine_chunks(self, chunks: List[str]) -> str:
        if not chunks:
            return ""

        if len(chunks) == 1:
            return chunks[0]

        combined = chunks[0]
        for chunk in chunks[1:]:
            test_combined = combined + "\n\n" + chunk
            if len(self.tokenizer.tokenize(test_combined)) <= self.effective_doc_length:
                combined = test_combined
            else:
                break

        return combined

    def _process_document_with_cache(self, doc: Dict, wrong_doc: Dict, processing_params: Dict) -> Optional[Dict]:
        doc_id = self._extract_id(doc['_id'])
        wrong_id = self._extract_id(wrong_doc['_id'])

        cache_key = self._get_cache_key(f"{doc_id}_{wrong_id}", processing_params)
        cached_result = self._load_from_cache(cache_key)

        if cached_result is not None:
            return cached_result

        cached_result = self._process_document_without_cache(doc, wrong_doc)
        if cached_result is not None:
            self._save_to_cache(cache_key, cached_result)

        return cached_result

    def _extract_id(self, value):
        if isinstance(value, dict) and '$oid' in value:
            return value['$oid']
        return str(value)

    def _process_dataset(self, json_path: str, split: str, train_ratio: float,
                         max_samples: Optional[int], seed: int) -> List[Dict[str, str]]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        doc_map = {self._extract_id(doc['_id']): doc for doc in data}
        processing_params = self._get_processing_params()

        all_samples = self._process_dataset_parallel(data, doc_map, processing_params)

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

    def clear_cache(self):
        if self.enable_cache and os.path.exists(self.cache_dir):
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict:
        if not self.enable_cache or not os.path.exists(self.cache_dir):
            return {"cache_enabled": False, "cache_files": 0, "cache_size_mb": 0}

        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in cache_files)

        return {
            "cache_enabled": True,
            "cache_files": len(cache_files),
            "cache_size_mb": round(total_size / (1024 * 1024), 2)
        }

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

    def get_processing_stats(self) -> Dict:
        with self.progress_lock:
            return self.progress_stats.copy()

    def set_num_workers(self, num_workers: int):
        self.num_workers = max(1, num_workers)
        logger.info(f"Updated number of workers to {self.num_workers}")

    def set_batch_size(self, batch_size: int):
        self.batch_size = max(1, batch_size)
        logger.info(f"Updated batch size to {self.batch_size}")


if __name__ == "__main__":
    dataset_processor = DatasetProcessor.preprocess_only(
        json_path='dataset.json',
        cache_dir='cache',
        force_refresh=False,
        num_workers=16,
        batch_size=50
    )

    print(f"Preprocessing completed. Cache stats: {dataset_processor.get_cache_stats()}")
    print(f"Processing stats: {dataset_processor.get_processing_stats()}")
