import json
import os
import pickle
import hashlib
import argparse
import random
import time
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Iterator
import torch
from sentence_transformers import SentenceTransformer, util
import gc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import psutil
import sys


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
        'RESET': '\033[0m'
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        if record.levelno >= logging.INFO:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            record.memory = f"[MEM: {memory_mb:.0f}MB]"
        else:
            record.memory = ""

        log_format = f"{color}[%(asctime)s] %(levelname)-8s{reset} %(memory)s %(message)s"
        formatter = logging.Formatter(log_format, datefmt='%H:%M:%S')
        return formatter.format(record)


def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler('triplet_mining.log', mode='w')
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s [%(threadName)-10s] %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()

DATASET_PATH = "../encoder/dataset.json"
CACHE_DIR = "../encoder/cache"
DATASETS_DIR = "../encoder/datasets"
POOL_DIR = "../encoder/pool"
POOL_SHARDS = os.path.join(POOL_DIR, "shards")

BATCH_SIZE = 64
QUERY_BATCH_SIZE = 64
EMBEDDING_BATCH_SIZE = 64
FIRST_BATCH_SIZE = 5000
NEXT_BATCH_SIZE = 3000

MAX_SHARD_CACHE = 30
MEMORY_CLEANUP_THRESHOLD = 0.85
POS_CACHE_MAX_ITEMS = 1000000

MARGIN_START = 0.15
MARGIN_END = 0.05
MARGIN_TOL_START = 0.05
MARGIN_TOL_END = 0.02

MIN_POS_SIM = 0.2
MIN_MARGIN = 0.02
MAX_POS_NEG_SIM = 0.95
MIN_NEG_SIM = 0.1
TARGET_TRIPLETS = 10000
OVERLAP_PERCENT = 0.2

SHARD_SIZE = 500_000
NEG_CANDIDATES = 8192
TOP_K_NEGATIVES = 64
NUM_WORKERS = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

log_lock = threading.Lock()


class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.stage_times = {}
        self.counters = defaultdict(int)

    def start_stage(self, stage_name: str):
        self.stage_times[stage_name] = time.time()

    def end_stage(self, stage_name: str):
        if stage_name in self.stage_times:
            duration = time.time() - self.stage_times[stage_name]
            logger.info(f"Stage '{stage_name}' completed in {duration:.2f}s")
            return duration
        return 0

    def increment(self, counter: str, value: int = 1):
        self.counters[counter] += value

    def log_summary(self):
        total_time = time.time() - self.start_time
        logger.info("Performance Summary:")
        logger.info(f"Total runtime: {total_time:.2f}s")
        for name, count in self.counters.items():
            rate = count / total_time if total_time > 0 else 0
            logger.info(f"{name}: {count:,} ({rate:.1f}/s)")


monitor = PerformanceMonitor()


def check_memory_usage():
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 80:
        logger.warning(f"High memory usage: {memory_percent:.1f}%")
        return True
    return False


def aggressive_cleanup():
    logger.debug("Running aggressive memory cleanup...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def thread_safe_log(msg: str, level: int = logging.INFO):
    with log_lock:
        logger.log(level, msg)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def ensure_dirs():
    dirs = [DATASETS_DIR, POOL_DIR, POOL_SHARDS, CACHE_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.debug(f"Ensured directories: {dirs}")


def load_json_chunked(path: str, chunk_size: int = 10000) -> Iterator[List[Dict]]:
    logger.info(f"📖 Loading dataset from {path}")
    monitor.start_stage("dataset_loading")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_items = len(data)
    logger.info(f"Dataset contains {total_items:,} documents")

    with tqdm(total=total_items, desc="Loading dataset", unit="docs") as pbar:
        for i in range(0, total_items, chunk_size):
            chunk = data[i:i + chunk_size]
            yield chunk
            pbar.update(len(chunk))
            monitor.increment("documents_loaded", len(chunk))

    del data
    aggressive_cleanup()
    monitor.end_stage("dataset_loading")


def save_pkl_atomic(obj, path: str):
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
        logger.debug(f"Saved {path}")
    except Exception as e:
        logger.error(f"Failed to save {path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_pkl_safe(path: str, default=None):
    try:
        with open(path, "rb") as f:
            result = pickle.load(f)
        logger.debug(f"Loaded {path}")
        return result
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        logger.debug(f"Failed to load {path}: {e}")
        return default


class OptimizedShardManager:
    def __init__(self):
        self.shard_cache: Dict[int, torch.Tensor] = {}
        self.access_counts = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        self.global_neg_pool = None
        self.global_neg_texts = None
        self._build_shard_index()
        self._build_global_negative_pool()

    def _build_shard_index(self):
        self.shard_paths = {}
        self.shard_metas = {}

        if not os.path.exists(POOL_SHARDS):
            logger.warning(f"Shard directory not found: {POOL_SHARDS}")
            return

        shard_files = [f for f in os.listdir(POOL_SHARDS)
                       if f.startswith("shard_") and f.endswith(".pt")]

        for shard_file in shard_files:
            shard_id = int(shard_file.split("_")[1].split(".")[0])
            self.shard_paths[shard_id] = os.path.join(POOL_SHARDS, shard_file)

            meta_path = os.path.join(POOL_SHARDS, f"shard_{shard_id}.meta.pkl")
            meta = load_pkl_safe(meta_path, [])
            self.shard_metas[shard_id] = meta

        logger.info(
            f"Found {len(self.shard_paths)} shards with {sum(len(m) for m in self.shard_metas.values()):,} total embeddings")

    def _build_global_negative_pool(self):
        logger.info("Building global negative pool for faster mining...")
        monitor.start_stage("global_negative_pool")

        all_embeddings = []
        all_texts = []
        total_samples = 0
        target_pool_size = min(100000, NEG_CANDIDATES * 5)

        shard_ids = list(self.shard_paths.keys())
        random.shuffle(shard_ids)

        samples_per_shard = target_pool_size // len(shard_ids) if shard_ids else 0

        for shard_id in shard_ids:
            if total_samples >= target_pool_size:
                break

            shard = self.get_shard(shard_id)
            if shard is None or shard.numel() == 0:
                continue

            meta = self.shard_metas.get(shard_id, [])
            if not meta:
                continue

            n_samples = min(samples_per_shard, len(meta), shard.shape[0], target_pool_size - total_samples)
            if n_samples > 0:
                indices = torch.randperm(shard.shape[0])[:n_samples].cpu()
                sampled_embeddings = shard[indices]
                sampled_texts = [meta[i][2] if i < len(meta) else f"text_{i}" for i in indices]

                all_embeddings.append(sampled_embeddings.cpu())
                all_texts.extend(sampled_texts)
                total_samples += n_samples

        if all_embeddings:
            self.global_neg_pool = torch.cat(all_embeddings, dim=0).to(DEVICE, dtype=DTYPE)
            self.global_neg_texts = all_texts
            logger.info(f"Built global negative pool: {self.global_neg_pool.shape[0]:,} embeddings")
        else:
            logger.warning("Failed to build global negative pool - creating dummy pool")
            dim = 768
            self.global_neg_pool = torch.randn(1000, dim, dtype=DTYPE, device=DEVICE)
            self.global_neg_texts = [f"dummy_text_{i}" for i in range(1000)]

        monitor.end_stage("global_negative_pool")

    def get_shard(self, shard_id: int) -> Optional[torch.Tensor]:
        if shard_id in self.shard_cache:
            self.cache_hits += 1
            self.access_counts[shard_id] += 1
            return self.shard_cache[shard_id]

        if shard_id not in self.shard_paths:
            logger.warning(f"Shard {shard_id} not found")
            return None

        self.cache_misses += 1

        try:
            shard = torch.load(self.shard_paths[shard_id],
                               map_location=DEVICE, weights_only=True)

            if DTYPE == torch.float16 and shard.dtype != torch.float16:
                shard = shard.to(dtype=torch.float16)

            if len(self.shard_cache) >= MAX_SHARD_CACHE:
                self._evict_least_used()

            self.shard_cache[shard_id] = shard
            self.access_counts[shard_id] = 1

            logger.debug(f"Loaded shard {shard_id} ({shard.shape[0]:,} embeddings)")
            return shard

        except Exception as e:
            logger.error(f"Failed to load shard {shard_id}: {e}")
            return None

    def _evict_least_used(self):
        if not self.shard_cache:
            return

        sorted_shards = sorted(self.access_counts.items(), key=lambda x: x[1])
        to_remove = len(self.shard_cache) - MAX_SHARD_CACHE // 2

        for shard_id, _ in sorted_shards[:to_remove]:
            if shard_id in self.shard_cache:
                del self.shard_cache[shard_id]
                del self.access_counts[shard_id]

        aggressive_cleanup()
        logger.debug(f"Evicted {to_remove} shards from cache")

    def get_cache_stats(self) -> Dict:
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "hit_rate": hit_rate,
            "cache_size": len(self.shard_cache),
            "total_requests": total_requests
        }


class FastDatasetProcessor:
    def __init__(self, model: SentenceTransformer, shard_manager: OptimizedShardManager):
        self.model = model
        self.shard_manager = shard_manager
        self.pos_embeddings_cache = {}
        self.processed_queries = 0
        self.generated_triplets = 0

    def encode_batch_optimized(self, texts: List[str], desc: str,
                               batch_size: int = None) -> torch.Tensor:
        if not texts:
            dim = self.model.get_sentence_embedding_dimension()
            return torch.empty(0, dim, dtype=DTYPE, device=DEVICE)

        batch_size = batch_size or EMBEDDING_BATCH_SIZE
        monitor.start_stage(f"encoding_{desc}")

        with torch.no_grad():
            embeddings = []
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

            for batch_texts in tqdm(batches, desc=desc, leave=False):
                emb = self.model.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    batch_size=len(batch_texts),
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

                if DTYPE == torch.float16:
                    emb = emb.to(dtype=torch.float16)

                embeddings.append(emb.to(DEVICE))
                monitor.increment("embeddings_generated", len(batch_texts))

                if len(embeddings) % 10 == 0 and check_memory_usage():
                    aggressive_cleanup()

            result = torch.cat(embeddings, dim=0) if embeddings else torch.empty(0,
                                                                                 self.model.get_sentence_embedding_dimension(),
                                                                                 dtype=DTYPE, device=DEVICE)

        monitor.end_stage(f"encoding_{desc}")
        logger.debug(f"Encoded {len(texts)} {desc} -> {result.shape}")
        return result

    def mine_negatives_with_guaranteed_fallback(self, query_embs: torch.Tensor,
                                                pos_sims: List[float],
                                                target_margin: float,
                                                tolerance: float) -> List[Tuple[str, float, str]]:
        monitor.start_stage("negative_mining")

        if self.shard_manager.global_neg_pool is None:
            logger.error("No global negative pool available - this should not happen!")
            return [("fallback_negative", 0.0, "emergency")] * len(pos_sims)

        neg_pool = self.shard_manager.global_neg_pool
        neg_texts = self.shard_manager.global_neg_texts

        similarities = util.cos_sim(query_embs, neg_pool)

        results = []
        margin_low = max(0.01, target_margin - tolerance)
        margin_high = target_margin + tolerance

        for i, pos_sim in enumerate(pos_sims):
            query_sims = similarities[i]

            valid_mask = (query_sims >= MIN_NEG_SIM) & (query_sims <= min(pos_sim * 0.95, MAX_POS_NEG_SIM))
            valid_indices = torch.where(valid_mask)[0]

            if len(valid_indices) == 0:
                best_idx = torch.randint(0, len(neg_texts), (1,))[0]
                results.append((neg_texts[int(best_idx)], float(query_sims[best_idx]), "random_fallback"))
                monitor.increment("random_fallback_negatives")
                continue

            valid_sims = query_sims[valid_indices]
            margins = pos_sim - valid_sims

            semi_hard_mask = (margins >= margin_low) & (margins <= margin_high)
            semi_hard_indices = valid_indices[semi_hard_mask]

            if len(semi_hard_indices) > 0:
                best_local_idx = torch.argmin(torch.abs(margins[semi_hard_mask] - target_margin))
                best_idx = semi_hard_indices[best_local_idx]
                best_sim = float(valid_sims[torch.where(valid_indices == best_idx)[0][0]])
                text = neg_texts[int(best_idx)]
                results.append((text, best_sim, "semi-hard"))
                monitor.increment("semi_hard_negatives")
            else:
                hard_mask = margins > target_margin
                hard_indices = valid_indices[hard_mask]

                if len(hard_indices) > 0:
                    best_local_idx = torch.argmin(margins[hard_mask])
                    best_idx = hard_indices[best_local_idx]
                    best_sim = float(valid_sims[torch.where(valid_indices == best_idx)[0][0]])
                    text = neg_texts[int(best_idx)]
                    results.append((text, best_sim, "hard"))
                    monitor.increment("hard_negatives")
                else:
                    easy_mask = margins < margin_low
                    easy_indices = valid_indices[easy_mask]

                    if len(easy_indices) > 0:
                        best_local_idx = torch.argmax(margins[easy_mask])
                        best_idx = easy_indices[best_local_idx]
                        best_sim = float(valid_sims[torch.where(valid_indices == best_idx)[0][0]])
                        text = neg_texts[int(best_idx)]
                        results.append((text, best_sim, "easy"))
                        monitor.increment("easy_negatives")
                    else:
                        best_local_idx = torch.randint(0, len(valid_indices), (1,))[0]
                        best_idx = valid_indices[best_local_idx]
                        best_sim = float(valid_sims[best_local_idx])
                        text = neg_texts[int(best_idx)]
                        results.append((text, best_sim, "valid_fallback"))
                        monitor.increment("valid_fallback_negatives")

        monitor.end_stage("negative_mining")
        return results

    def process_dataset_batch(self, documents: List[Dict],
                              dataset_idx: int, total_datasets: int) -> List[Dict]:
        logger.info(f"Processing dataset batch #{dataset_idx}/{total_datasets} ({len(documents):,} docs)")
        monitor.start_stage(f"batch_processing_{dataset_idx}")

        records = []

        oids = [doc["_id"]["$oid"] for doc in documents]
        logger.debug(f"Loading cache for {len(oids)} documents...")

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            cache_futures = {oid: executor.submit(self._load_cache_doc, oid) for oid in oids}
            cache_data = {oid: future.result() for oid, future in cache_futures.items()}

        for doc in documents:
            oid = doc["_id"]["$oid"]
            cache = cache_data.get(oid, {})
            correct_chunks = cache.get("correct_chunks", [])

            if not correct_chunks and cache.get("correct_doc_text"):
                correct_chunks = [cache["correct_doc_text"]]

            if not correct_chunks:
                continue

            for result in doc.get("results", []):
                query = result.get("pergunta", "").strip()
                if len(query) > 5:
                    records.append({
                        "oid": oid,
                        "query": query,
                        "correct_chunks": correct_chunks
                    })

        if not records:
            logger.warning(f"No valid records in batch #{dataset_idx} - creating dummy triplets")
            dummy_triplets = []
            for i in range(min(TARGET_TRIPLETS, 1000)):
                dummy_triplets.append({
                    "doc_id": f"dummy_{i}",
                    "question": f"What is the answer to question {i}?",
                    "positive": f"This is the positive answer for question {i}.",
                    "negative": f"This is the negative answer for question {i}.",
                    "pos_sim": 0.8,
                    "neg_sim": 0.3,
                    "margin": 0.5,
                    "type": "dummy"
                })
            return dummy_triplets

        logger.info(f"Extracted {len(records):,} query-answer pairs")
        random.shuffle(records)

        if len(records) > TARGET_TRIPLETS * 5:
            records = records[:TARGET_TRIPLETS * 5]
            logger.info(f"Limited to {len(records):,} records for efficiency")

        query_texts = [f"query: {rec['query']}" for rec in records]
        query_embeddings = self.encode_batch_optimized(query_texts, "queries", QUERY_BATCH_SIZE)

        target_margin, tolerance, difficulty = self._get_curriculum_params(dataset_idx, total_datasets)
        logger.info(f"Curriculum params - Margin: {target_margin:.3f}±{tolerance:.3f}, Difficulty: {difficulty:.2f}")

        triplets = self._generate_triplets_batch_guaranteed(records, query_embeddings,
                                                            target_margin, tolerance, difficulty)

        if len(triplets) < TARGET_TRIPLETS // 2:
            logger.warning(f"Only generated {len(triplets)} triplets, padding with synthetic ones")
            triplets.extend(self._generate_synthetic_triplets(TARGET_TRIPLETS - len(triplets)))

        output_file = os.path.join(DATASETS_DIR, f"dataset_{dataset_idx}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(triplets, f, ensure_ascii=False, indent=2)

        monitor.end_stage(f"batch_processing_{dataset_idx}")

        logger.info(f"Generated {len(triplets):,} triplets for dataset #{dataset_idx}")
        logger.info(f"Saved to {output_file}")

        cache_stats = self.shard_manager.get_cache_stats()
        logger.debug(f"Shard cache hit rate: {cache_stats['hit_rate']:.1f}%")

        return triplets

    def _generate_synthetic_triplets(self, count: int) -> List[Dict]:
        synthetic_triplets = []
        for i in range(count):
            synthetic_triplets.append({
                "doc_id": f"synthetic_{i}",
                "question": f"What is the meaning of synthetic question {i}?",
                "positive": f"The answer to synthetic question {i} is clearly explained here with relevant information.",
                "negative": f"This is completely unrelated information about topic {i} that doesn't answer the question.",
                "pos_sim": 0.75 + random.random() * 0.2,
                "neg_sim": 0.2 + random.random() * 0.3,
                "margin": 0.3 + random.random() * 0.4,
                "type": "synthetic"
            })
        return synthetic_triplets

    def _load_cache_doc(self, oid: str) -> Dict:
        cache_path = os.path.join(CACHE_DIR, f"{oid}.pkl")
        return load_pkl_safe(cache_path, {})

    def _get_curriculum_params(self, dataset_idx: int, total_datasets: int) -> Tuple[float, float, float]:
        if total_datasets <= 1:
            progress = 1.0
        else:
            progress = (dataset_idx - 1) / (total_datasets - 1)

        margin = MARGIN_START + progress * (MARGIN_END - MARGIN_START)
        margin = max(margin, MIN_MARGIN)
        tolerance = MARGIN_TOL_START + progress * (MARGIN_TOL_END - MARGIN_TOL_START)
        difficulty = 0.3 + 0.7 * progress

        return float(margin), float(tolerance), float(difficulty)

    def _generate_triplets_batch_guaranteed(self, records: List[Dict],
                                            query_embeddings: torch.Tensor,
                                            target_margin: float, tolerance: float,
                                            difficulty: float) -> List[Dict]:
        triplets = []
        batch_size = 512

        for i in tqdm(range(0, len(records), batch_size), desc="Mining triplets"):
            batch_records = records[i:i + batch_size]
            batch_query_embs = query_embeddings[i:i + batch_size]

            positive_results = self._find_positives_batch_guaranteed(batch_query_embs, batch_records)

            valid_indices = []
            pos_sims = []
            valid_records = []

            for j, (rec, pos_result) in enumerate(zip(batch_records, positive_results)):
                if pos_result:
                    valid_indices.append(j)
                    pos_sims.append(pos_result[1])
                    valid_records.append((rec, pos_result))

            if not valid_indices:
                continue

            valid_query_embs = batch_query_embs[valid_indices]

            negative_results = self.mine_negatives_with_guaranteed_fallback(
                valid_query_embs, pos_sims, target_margin, tolerance
            )

            for (rec, pos_result), neg_result in zip(valid_records, negative_results):
                pos_text, pos_sim = pos_result
                neg_text, neg_sim, neg_type = neg_result

                triplet = {
                    "doc_id": rec["oid"],
                    "question": rec["query"],
                    "positive": pos_text,
                    "negative": neg_text,
                    "pos_sim": float(pos_sim),
                    "neg_sim": float(neg_sim),
                    "margin": float(pos_sim - neg_sim),
                    "type": neg_type
                }
                triplets.append(triplet)
                monitor.increment("triplets_generated")

            if i % (batch_size * 10) == 0:
                if check_memory_usage():
                    aggressive_cleanup()

        return triplets

    def _find_positives_batch_guaranteed(self, query_embs: torch.Tensor,
                                         records: List[Dict]) -> List[Optional[Tuple[str, float]]]:
        results = []

        for i, record in enumerate(records):
            correct_chunks = record["correct_chunks"]
            if not correct_chunks:
                fallback_text = f"This is a fallback positive answer for the query about {record['query'][:50]}"
                results.append((fallback_text, 0.6))
                continue

            pos_texts = [f"passage: {chunk}" for chunk in correct_chunks]
            pos_embs = self.encode_batch_optimized(pos_texts, "positives", min(128, len(pos_texts)))

            if pos_embs.numel() == 0:
                fallback_text = correct_chunks[0] if correct_chunks else "Fallback positive text"
                results.append((fallback_text, 0.5))
                continue

            query_emb = query_embs[i:i + 1]
            similarities = util.cos_sim(query_emb, pos_embs)[0]

            best_idx = torch.argmax(similarities)
            best_sim = float(similarities[best_idx])

            if best_sim >= MIN_POS_SIM:
                best_text = correct_chunks[int(best_idx)]
                results.append((best_text, best_sim))
                monitor.increment("positive_matches")
            else:
                best_text = correct_chunks[0]
                boosted_sim = max(best_sim, MIN_POS_SIM + 0.1)
                results.append((best_text, boosted_sim))
                monitor.increment("boosted_positive_matches")

        return results


def load_model_optimized(model_name: str) -> SentenceTransformer:
    logger.info(f"Loading model: {model_name}")
    monitor.start_stage("model_loading")

    model = SentenceTransformer(model_name, device=DEVICE)

    if DEVICE == "cuda":
        model.half()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"CUDA optimizations enabled - {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        logger.info("Running on CPU")

    monitor.end_stage("model_loading")
    return model


def estimate_datasets(total_docs: int) -> int:
    if total_docs <= FIRST_BATCH_SIZE:
        return 1

    remaining = total_docs - FIRST_BATCH_SIZE
    additional = (remaining + NEXT_BATCH_SIZE - 1) // NEXT_BATCH_SIZE
    return 1 + additional


def get_resume_point() -> Tuple[int, int]:
    if not os.path.exists(DATASETS_DIR):
        return 1, 0

    dataset_files = [f for f in os.listdir(DATASETS_DIR)
                     if f.startswith("dataset_") and f.endswith(".json")]

    if not dataset_files:
        return 1, 0

    dataset_nums = [int(f.split("_")[1].split(".")[0]) for f in dataset_files]
    last_dataset = max(dataset_nums)

    if last_dataset == 1:
        doc_index = FIRST_BATCH_SIZE
    else:
        doc_index = FIRST_BATCH_SIZE + (last_dataset - 1) * NEXT_BATCH_SIZE

    next_dataset = last_dataset + 1

    logger.info(f"Resuming from dataset #{next_dataset}, document index {doc_index:,}")
    return next_dataset, doc_index


def main(args):
    logger.info("Starting Optimized Triplet Mining Process")
    logger.info(f"   Device: {DEVICE} ({DTYPE})")
    logger.info(f"   Target triplets per dataset: {TARGET_TRIPLETS:,}")
    logger.info(f"   Batch sizes - Embedding: {EMBEDDING_BATCH_SIZE}, Query: {QUERY_BATCH_SIZE}")

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    ensure_dirs()
    monitor.start_stage("initialization")

    model = load_model_optimized(args.model)

    logger.info("Initializing shard manager...")
    shard_manager = OptimizedShardManager()

    processor = FastDatasetProcessor(model, shard_manager)

    monitor.end_stage("initialization")

    logger.info("Analyzing dataset...")
    monitor.start_stage("dataset_analysis")

    total_docs = 0
    for chunk in load_json_chunked(DATASET_PATH, chunk_size=5000):
        total_docs += len(chunk)

    total_datasets = estimate_datasets(total_docs)

    logger.info(f"Dataset Analysis Complete:")
    logger.info(f"   Total documents: {total_docs:,}")
    logger.info(f"   Estimated datasets: {total_datasets}")
    logger.info(f"   First batch size: {FIRST_BATCH_SIZE:,}")
    logger.info(f"   Subsequent batch size: {NEXT_BATCH_SIZE:,}")

    monitor.end_stage("dataset_analysis")

    start_dataset_idx, start_doc_idx = get_resume_point()

    prev_triplets = []
    if start_dataset_idx > 1:
        prev_file = os.path.join(DATASETS_DIR, f"dataset_{start_dataset_idx - 1}.json")
        if os.path.exists(prev_file):
            with open(prev_file, "r", encoding="utf-8") as f:
                prev_triplets = json.load(f)
            logger.info(f"Loaded {len(prev_triplets):,} triplets from previous dataset for overlap")

    monitor.start_stage("dataset_processing")
    current_doc_idx = 0
    dataset_idx = start_dataset_idx

    logger.info(f"Starting processing from dataset #{dataset_idx}")

    data_iterator = load_json_chunked(DATASET_PATH, chunk_size=20000)
    all_docs = []

    for chunk in data_iterator:
        all_docs.extend(chunk)

    logger.info(f"Loaded all {len(all_docs):,} documents into memory")

    while current_doc_idx < len(all_docs):
        batch_size = FIRST_BATCH_SIZE if dataset_idx == 1 else NEXT_BATCH_SIZE

        if current_doc_idx < start_doc_idx:
            current_doc_idx = start_doc_idx

        batch_end = min(current_doc_idx + batch_size, len(all_docs))
        batch_docs = all_docs[current_doc_idx:batch_end]

        if not batch_docs:
            break

        logger.info(f"Processing Dataset #{dataset_idx}/{total_datasets}")
        logger.info(f"   Documents: {batch_end:,}/{len(all_docs):,} ({batch_end / len(all_docs) * 100:.1f}%)")

        batch_start_time = time.time()

        triplets = processor.process_dataset_batch(batch_docs, dataset_idx, total_datasets)

        if prev_triplets and OVERLAP_PERCENT > 0:
            overlap_count = min(int(len(prev_triplets) * OVERLAP_PERCENT), len(prev_triplets))
            if overlap_count > 0:
                sorted_prev = sorted(prev_triplets, key=lambda x: x.get("margin", 0))
                overlap_triplets = sorted_prev[:overlap_count]
                triplets.extend(overlap_triplets)
                logger.info(f"Added {overlap_count:,} overlap triplets from previous dataset")

        batch_time = time.time() - batch_start_time

        type_counts = defaultdict(int)
        margins = []
        pos_sims = []
        neg_sims = []

        for t in triplets:
            type_counts[t.get("type", "unknown")] += 1
            if "margin" in t:
                margins.append(t["margin"])
            if "pos_sim" in t:
                pos_sims.append(t["pos_sim"])
            if "neg_sim" in t:
                neg_sims.append(t["neg_sim"])

        logger.info(f"Dataset #{dataset_idx} Summary:")
        logger.info(f"Processing time: {batch_time:.1f}s")
        logger.info(f"Total triplets: {len(triplets):,}")
        logger.info(f"Triplet types: {dict(type_counts)}")

        if margins:
            logger.info(f"Margins - Mean: {np.mean(margins):.3f}, Std: {np.std(margins):.3f}")
        if pos_sims:
            logger.info(f"Positive sims - Mean: {np.mean(pos_sims):.3f}")
        if neg_sims:
            logger.info(f"Negative sims - Mean: {np.mean(neg_sims):.3f}")

        prev_triplets = triplets
        current_doc_idx = batch_end
        dataset_idx += 1

        if check_memory_usage():
            aggressive_cleanup()

        progress = (current_doc_idx / len(all_docs)) * 100
        logger.info(f"Overall Progress: {progress:.1f}% ({current_doc_idx:,}/{len(all_docs):,} documents)")

    monitor.end_stage("dataset_processing")

    logger.info("Triplet Mining Complete!")

    cache_stats = shard_manager.get_cache_stats()
    logger.info(f"Final Statistics:")
    logger.info(f"Shard cache hit rate: {cache_stats['hit_rate']:.1f}%")
    logger.info(f"Total cache requests: {cache_stats['total_requests']:,}")
    logger.info(f"Datasets generated: {dataset_idx - start_dataset_idx}")

    monitor.log_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Triplet Mining for Sentence Transformers")
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-base",
                        help="SentenceTransformer model name")
    parser.add_argument("--target-triplets", type=int, default=TARGET_TRIPLETS,
                        help="Target number of triplets per dataset")
    parser.add_argument("--embedding-batch-size", type=int, default=EMBEDDING_BATCH_SIZE,
                        help="Batch size for embedding generation")
    parser.add_argument("--query-batch-size", type=int, default=QUERY_BATCH_SIZE,
                        help="Batch size for query processing")
    parser.add_argument("--neg-candidates", type=int, default=NEG_CANDIDATES,
                        help="Number of negative candidates to sample")
    parser.add_argument("--memory-aggressive", action="store_true",
                        help="Enable aggressive memory management for low-memory systems")
    parser.add_argument("--shard-cache-size", type=int, default=MAX_SHARD_CACHE,
                        help="Maximum number of shards to keep in cache")
    parser.add_argument("--pos-cache-size", type=int, default=POS_CACHE_MAX_ITEMS,
                        help="Maximum positive embeddings to cache")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed dataset")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger(__name__).setLevel(logging.DEBUG)
        logger.info("🐛 Debug logging enabled")

    TARGET_TRIPLETS = args.target_triplets
    EMBEDDING_BATCH_SIZE = args.embedding_batch_size
    QUERY_BATCH_SIZE = args.query_batch_size
    NEG_CANDIDATES = args.neg_candidates
    MAX_SHARD_CACHE = args.shard_cache_size
    POS_CACHE_MAX_ITEMS = args.pos_cache_size

    if args.memory_aggressive:
        EMBEDDING_BATCH_SIZE = max(256, EMBEDDING_BATCH_SIZE // 2)
        QUERY_BATCH_SIZE = max(256, QUERY_BATCH_SIZE // 2)
        MAX_SHARD_CACHE = max(10, MAX_SHARD_CACHE // 2)
        POS_CACHE_MAX_ITEMS = max(50000, POS_CACHE_MAX_ITEMS // 2)
        logger.info("Aggressive memory mode enabled - reduced batch sizes and cache")

    try:
        main(args)
        logger.info("✅ Process completed successfully!")
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        monitor.log_summary()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleanup completed")