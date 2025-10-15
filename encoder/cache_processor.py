import os
import json
import pickle
from typing import Dict, List, Optional, Tuple
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import ijson
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CacheProcessor:
    def __init__(
        self,
        tokenizer=None,
        max_length=512,
        split="all",
        max_samples=None,
        seed=42,
        json_path="dataset.json",
        cache_dir="cache",
        enable_cache=True,
        num_workers=4,
        batch_size=100,
        chunk_overlap=50,
        use_chunking_threshold=400,
        min_chunk_tokens=100,
        preserve_structure=True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained("./models/albertina-1b5") if tokenizer is None else tokenizer
        self.max_length = max_length
        self.effective_doc_length = max_length - 20
        self.chunk_overlap = chunk_overlap
        self.use_chunking_threshold = use_chunking_threshold
        self.min_chunk_tokens = min_chunk_tokens
        self.preserve_structure = preserve_structure
        self.cache_dir = cache_dir
        self.enable_cache = enable_cache
        self.json_path = json_path
        self.split = split
        self.max_samples = max_samples
        self.seed = seed
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.cache_lock = Lock()
        self.progress_lock = Lock()
        self.progress_stats = {"processed": 0, "cache_hits": 0, "new_processed": 0, "errors": 0}

        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cached_keys = set()
            if os.path.exists(self.cache_dir):
                self.cached_keys.update([f[:-4] for f in os.listdir(self.cache_dir) if f.endswith(".pkl")])
        else:
            self.cached_keys = set()

    @classmethod
    def preprocess_only(
        cls,
        json_path: str,
        cache_dir: str = "cache",
        max_length: int = 512,
        force_refresh: bool = False,
        num_workers: int = 4,
        batch_size: int = 100,
        chunk_overlap: int = 50,
        use_chunking_threshold: int = 400,
        min_chunk_tokens: int = 100,
        preserve_structure: bool = True,
    ):
        processor = cls(
            tokenizer=AutoTokenizer.from_pretrained("./models/albertina-1b5"),
            max_length=max_length,
            split="all",
            max_samples=None,
            seed=42,
            json_path=json_path,
            cache_dir=cache_dir,
            enable_cache=True,
            num_workers=num_workers,
            batch_size=batch_size,
            chunk_overlap=chunk_overlap,
            use_chunking_threshold=use_chunking_threshold,
            min_chunk_tokens=min_chunk_tokens,
            preserve_structure=preserve_structure,
        )

        if force_refresh:
            processor.clear_cache()

        processor._preprocess_all_documents()
        return processor

    def _create_worker_tokenizer(self):
        return AutoTokenizer.from_pretrained("./models/albertina-1b5")

    def _should_chunk_fast(self, text: str) -> bool:
        tokens = self.tokenizer.tokenize(text)
        return len(tokens) > self.use_chunking_threshold

    def _tokenize_and_chunk_fast(self, text: str, metadata: str = "") -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) <= self.effective_doc_length:
            return [text]
        chunks = []
        stride = 32
        start = 0
        while start < len(tokens):
            end = min(start + self.effective_doc_length, len(tokens))
            chunk_tokens = tokens[start:end]
            if len(chunk_tokens) >= self.min_chunk_tokens:
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                if metadata and start == 0:
                    chunk_text = f"{metadata}\n\n{chunk_text}"
                chunks.append(chunk_text)
            if end >= len(tokens):
                break
            start += stride
        return chunks if chunks else [text]

    def _extract_id(self, value):
        if isinstance(value, dict) and "$oid" in value:
            return value["$oid"]
        return str(value)

    def _get_cache_key(self, doc_id: str) -> str:
        return doc_id

    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        if not self.enable_cache or cache_key not in self.cached_keys:
            return None
        try:
            with open(self._get_cache_path(cache_key), "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_to_cache(self, cache_key: str, data: Dict) -> None:
        if not self.enable_cache:
            return
        cache_path = self._get_cache_path(cache_key)
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
        self.cached_keys.add(cache_key)

    def _extract_document_metadata(self, doc: Dict) -> str:
        parts = []
        if "numero" in doc:
            parts.append(f"Documento nº {doc['numero']}")
        if "ementa" in doc:
            parts.append(f"Ementa: {doc['ementa'][:100]}...")
        if "tribunal" in doc:
            parts.append(f"Tribunal: {doc['tribunal']}")
        if "data" in doc:
            parts.append(f"Data: {doc['data']}")
        return " | ".join(parts) if parts else ""

    def _process_positive_only(self, doc: Dict) -> Optional[Dict]:
        text = doc.get("Texto Integral", "")
        if not text:
            return None
        metadata = self._extract_document_metadata(doc)
        if self._should_chunk_fast(text):
            chunks = self._tokenize_and_chunk_fast(text, metadata)
            return {"correct_chunks": chunks, "is_chunked": True, "correct_doc_text": text}
        else:
            return {"correct_chunks": [text], "is_chunked": False, "correct_doc_text": text}

    def _create_document_batches(self, docs: List[Dict]) -> List[List[Dict]]:
        return [docs[i : i + self.batch_size] for i in range(0, len(docs), self.batch_size)]

    def _process_document_batch(self, doc_batch: List[Dict]) -> List[Optional[Dict]]:
        local_tokenizer = self._create_worker_tokenizer()
        original_tokenizer = self.tokenizer
        self.tokenizer = local_tokenizer

        results = []
        local_stats = {"processed": 0, "cache_hits": 0, "new_processed": 0, "errors": 0}

        try:
            for doc in doc_batch:
                try:
                    doc_id = self._extract_id(doc["_id"])
                    cache_key = self._get_cache_key(doc_id)
                    if cache_key in self.cached_keys:
                        local_stats["cache_hits"] += 1
                        results.append({"cached": True, "doc_id": doc_id})
                    else:
                        processed = self._process_positive_only(doc)
                        if processed is not None:
                            with self.cache_lock:
                                self._save_to_cache(cache_key, processed)
                            local_stats["new_processed"] += 1
                            results.append(processed)
                        else:
                            results.append(None)
                    local_stats["processed"] += 1
                except Exception:
                    local_stats["errors"] += 1
                    results.append(None)
        finally:
            self.tokenizer = original_tokenizer

        with self.progress_lock:
            for k, v in local_stats.items():
                self.progress_stats[k] += v

        return results

    def _preprocess_all_documents(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = []
        existing_cache = 0
        for doc in data:
            if not isinstance(doc, dict):
                continue
            if "Texto Integral" not in doc:
                continue
            doc_id = self._extract_id(doc.get("_id", ""))
            cache_key = self._get_cache_key(doc_id)
            if self.enable_cache and cache_key in self.cached_keys:
                existing_cache += 1
                continue
            docs.append(doc)

        if not docs:
            return

        batches = self._create_document_batches(docs)
        with self.progress_lock:
            self.progress_stats = {"processed": 0, "cache_hits": existing_cache, "new_processed": 0, "errors": 0}

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._process_document_batch, batch): i for i, batch in enumerate(batches)}
            for _ in as_completed(futures):
                pass

    def clear_cache(self):
        if self.enable_cache and os.path.exists(self.cache_dir):
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cached_keys.clear()

    def get_cache_stats(self) -> Dict:
        if not self.enable_cache or not os.path.exists(self.cache_dir):
            return {"cache_enabled": False, "cache_files": 0, "cache_size_mb": 0}
        files = [f for f in os.listdir(self.cache_dir) if f.endswith(".pkl")]
        total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in files)
        return {"cache_enabled": True, "cache_files": len(files), "cache_size_mb": round(total_size / (1024 * 1024), 2)}

    def get_processing_stats(self) -> Dict:
        with self.progress_lock:
            return self.progress_stats.copy()

    def set_num_workers(self, num_workers: int):
        self.num_workers = max(1, num_workers)

    def set_batch_size(self, batch_size: int):
        self.batch_size = max(1, batch_size)


if __name__ == "__main__":
    dataset_processor = CacheProcessor.preprocess_only(
        json_path="dataset.json",
        cache_dir="cache",
        force_refresh=False,
        num_workers=16,
        batch_size=50,
    )
    print(f"Cache stats: {dataset_processor.get_cache_stats()}")
    print(f"Processing stats: {dataset_processor.get_processing_stats()}")
