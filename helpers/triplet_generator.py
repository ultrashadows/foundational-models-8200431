import json
import os
import pickle
import argparse
import random
import time
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import torch
from sentence_transformers import SentenceTransformer, util
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import psutil
import sys
from transformers import AutoTokenizer
import ijson


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
POOL_PATH = "../encoder/pool.json"
POOL_CACHE_DIR = "../encoder/pool_cache"
TRIPLET_CHECKPOINT_DIR = "../encoder/triplet_checkpoints"

BATCH_SIZE = 64
QUERY_BATCH_SIZE = 64
EMBEDDING_BATCH_SIZE = 64
FIRST_BATCH_SIZE = 125000
NEXT_BATCH_SIZE = 0

MEMORY_CLEANUP_THRESHOLD = 0.85
POS_CACHE_MAX_ITEMS = 1000000

MARGIN_START = 0.25
MARGIN_END = 0.10
MARGIN_TOL_START = 0.10
MARGIN_TOL_END = 0.05

MIN_POS_SIM = 0.25
MIN_MARGIN = 0.12
MAX_POS_NEG_SIM = 0.88
MIN_NEG_SIM = 0.20
TARGET_TRIPLETS = 125000
OVERLAP_PERCENT = 0.0

NEG_CANDIDATES = 16384
TOP_K_NEGATIVES = 128
NUM_WORKERS = 8

MAX_CHUNK_USAGE = 3
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
MIN_CHUNK_TOKENS = 150
CHECKPOINT_EVERY = 5000
SIMILARITY_BATCH_SIZE = 512
MAX_POOL_SIZE = 2000000

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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def thread_safe_log(msg: str, level: int = logging.INFO):
    with log_lock:
        logger.log(level, msg)


def ensure_dirs():
    dirs = [DATASETS_DIR, CACHE_DIR, POOL_CACHE_DIR, TRIPLET_CHECKPOINT_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_pkl_safe(path: str, default=None):
    try:
        with open(path, "rb") as f:
            result = pickle.load(f)
        return result
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        return default


def save_pkl_atomic(obj, path: str):
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    except Exception as e:
        logger.error(f"Failed to save {path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


class PoolChunker:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.effective_doc_length = CHUNK_SIZE - 20

    def chunk_document(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)

        if len(tokens) <= self.effective_doc_length:
            return [text]

        chunks = []
        stride = self.effective_doc_length - CHUNK_OVERLAP
        start = 0

        while start < len(tokens):
            end = min(start + self.effective_doc_length, len(tokens))
            chunk_tokens = tokens[start:end]

            if len(chunk_tokens) >= MIN_CHUNK_TOKENS:
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                chunks.append(chunk_text)

            if end >= len(tokens):
                break

            start += stride

        return chunks if chunks else [text]


class NegativePoolManager:
    def __init__(self, pool_path: str, model: SentenceTransformer, tokenizer):
        self.pool_path = pool_path
        self.model = model
        self.tokenizer = tokenizer
        self.chunker = PoolChunker(tokenizer)
        self.global_neg_pool = None
        self.global_neg_texts = None
        self.chunk_usage_counts = None
        self._load_or_build_pool()

    def _load_or_build_pool(self):
        cache_path = os.path.join(POOL_CACHE_DIR, "pool_embeddings.pt")
        metadata_path = os.path.join(POOL_CACHE_DIR, "pool_metadata.pkl")

        pool_index_json = os.path.join(POOL_CACHE_DIR, "pool_index.json")
        pool_index_pkl = os.path.join(POOL_CACHE_DIR, "pool_index.pkl")

        if os.path.exists(pool_index_json) or os.path.exists(pool_index_pkl):
            logger.info("Found sharded pool cache, loading from shards...")
            self._load_from_shards()
            return

        if os.path.exists(cache_path) and os.path.exists(metadata_path):
            logger.info("Found existing pool cache, loading...")
            try:
                logger.info(f"Loading embeddings from {cache_path}...")
                cache = torch.load(cache_path, map_location='cpu', weights_only=True)

                logger.info(f"Loading metadata from {metadata_path}...")
                metadata = load_pkl_safe(metadata_path, {})

                logger.info("Moving embeddings to device...")
                self.global_neg_pool = cache['embeddings'].to(DEVICE, dtype=DTYPE)
                self.global_neg_texts = metadata['texts']
                self.chunk_usage_counts = [0] * len(self.global_neg_texts)

                logger.info(f"Successfully loaded {self.global_neg_pool.shape[0]:,} cached chunk embeddings")
                logger.info(f"Pool ready with {len(self.global_neg_texts):,} negative chunks")
                return
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                logger.info("Will rebuild pool from scratch...")

        logger.info(f"No cache found. Building pool from {self.pool_path}")
        logger.info("This will take a while but only needs to be done once...")
        monitor.start_stage("pool_building")

        if not os.path.exists(self.pool_path):
            logger.error(f"Pool file not found: {self.pool_path}")
            self._create_dummy_pool()
            monitor.end_stage("pool_building")
            return

        logger.info("Streaming pool.json with ijson (this may take a few minutes for 7GB file)...")
        all_chunks = []
        doc_count = 0

        with open(self.pool_path, "rb") as f:
            parser = ijson.items(f, "item")

            for doc in tqdm(parser, desc="Streaming and chunking pool documents", unit=" docs"):
                text = doc.get("Texto Integral", "")
                if text and len(text.strip()) > 50:
                    chunks = self.chunker.chunk_document(text.strip())
                    all_chunks.extend(chunks)
                    doc_count += 1

                if doc_count % 10000 == 0 and check_memory_usage():
                    logger.info(f"Processed {doc_count:,} docs, created {len(all_chunks):,} chunks so far")
                    aggressive_cleanup()

        if not all_chunks:
            logger.error("No valid chunks created from pool!")
            self._create_dummy_pool()
            monitor.end_stage("pool_building")
            return

        logger.info(f"Created {len(all_chunks):,} chunks from {doc_count:,} pool documents")

        logger.info("Encoding pool chunks...")
        embeddings = self._encode_pool_texts(all_chunks)

        self.global_neg_pool = embeddings.to(DEVICE, dtype=DTYPE)
        self.global_neg_texts = all_chunks
        self.chunk_usage_counts = [0] * len(all_chunks)

        logger.info(f"Saving pool cache to {POOL_CACHE_DIR}...")
        try:
            torch.save({
                'embeddings': embeddings.cpu()
            }, cache_path)

            save_pkl_atomic({
                'texts': all_chunks
            }, metadata_path)

            logger.info(f"Pool cache saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

        logger.info(f"Built negative pool: {self.global_neg_pool.shape[0]:,} chunk embeddings")
        monitor.end_stage("pool_building")

    def _load_from_shards(self):
        logger.info("Loading pool from sharded cache...")
        monitor.start_stage("loading_sharded_pool")

        pool_index_path = os.path.join(POOL_CACHE_DIR, "pool_index.json")
        if not os.path.exists(pool_index_path):
            pool_index_path = os.path.join(POOL_CACHE_DIR, "pool_index.pkl")

        if pool_index_path.endswith('.json'):
            with open(pool_index_path, 'r') as f:
                index = json.load(f)
        else:
            index = load_pkl_safe(pool_index_path, {})

        num_shards = index.get('num_shards', 238)
        logger.info(f"Found {num_shards} shards")

        all_embeddings = []
        all_texts = []

        for shard_id in tqdm(range(1, num_shards + 1), desc="Loading pool shards"):
            embed_path = os.path.join(POOL_CACHE_DIR, f"pool_embeds_{shard_id:05d}.pt")
            text_path = os.path.join(POOL_CACHE_DIR, f"pool_texts_{shard_id:05d}.pkl")

            if not os.path.exists(embed_path) or not os.path.exists(text_path):
                continue

            try:
                shard_data = torch.load(embed_path, map_location='cpu', weights_only=False)
                shard_texts = load_pkl_safe(text_path, [])

                if isinstance(shard_data, dict):
                    if 'embeddings' in shard_data:
                        shard_embeds = shard_data['embeddings']
                    elif 'emb' in shard_data:
                        shard_embeds = shard_data['emb']
                    else:
                        shard_embeds = list(shard_data.values())[0]
                else:
                    shard_embeds = shard_data

                all_embeddings.append(shard_embeds.cpu())
                all_texts.extend(shard_texts)

            except Exception as e:
                logger.error(f"Failed to load shard {shard_id}: {e}")
                continue

        if not all_embeddings:
            logger.error("No shards loaded successfully!")
            self._create_dummy_pool()
            monitor.end_stage("loading_sharded_pool")
            return

        logger.info(f"Concatenating {len(all_embeddings)} shards...")
        embeddings = torch.cat(all_embeddings, dim=0)

        total_chunks = embeddings.shape[0]
        logger.info(f"Total chunks: {total_chunks:,}")

        if total_chunks > MAX_POOL_SIZE:
            logger.info(f"Sampling {MAX_POOL_SIZE:,} chunks from {total_chunks:,} for faster similarity search")
            indices = torch.randperm(total_chunks)[:MAX_POOL_SIZE]
            embeddings = embeddings[indices]
            all_texts = [all_texts[i] for i in indices.tolist()]

        logger.info("Moving embeddings to device...")
        self.global_neg_pool = embeddings.to(DEVICE, dtype=DTYPE)
        self.global_neg_texts = all_texts
        self.chunk_usage_counts = [0] * len(all_texts)

        del all_embeddings
        aggressive_cleanup()

        logger.info(f"Successfully loaded {self.global_neg_pool.shape[0]:,} chunk embeddings")
        logger.info(f"Pool ready with {len(self.global_neg_texts):,} negative chunks")
        monitor.end_stage("loading_sharded_pool")

    def _encode_pool_texts(self, texts: List[str]) -> torch.Tensor:
        batch_size = EMBEDDING_BATCH_SIZE
        embeddings = []

        with torch.no_grad():
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

            for batch_texts in tqdm(batches, desc="Encoding pool chunks"):
                emb = self.model.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    batch_size=len(batch_texts),
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

                if DTYPE == torch.float16:
                    emb = emb.to(dtype=torch.float16)

                embeddings.append(emb.cpu())
                monitor.increment("pool_embeddings_generated", len(batch_texts))

                if len(embeddings) % 10 == 0 and check_memory_usage():
                    aggressive_cleanup()

        return torch.cat(embeddings, dim=0)

    def _create_dummy_pool(self):
        logger.warning("Creating dummy negative pool")
        dim = 768
        self.global_neg_pool = torch.randn(1000, dim, dtype=DTYPE, device=DEVICE)
        self.global_neg_texts = [f"dummy_text_{i}" for i in range(1000)]
        self.chunk_usage_counts = [0] * 1000

    def get_available_indices(self) -> torch.Tensor:
        available = [i for i, count in enumerate(self.chunk_usage_counts) if count < MAX_CHUNK_USAGE]
        return torch.tensor(available, dtype=torch.long)

    def increment_usage(self, indices: List[int]):
        for idx in indices:
            if 0 <= idx < len(self.chunk_usage_counts):
                self.chunk_usage_counts[idx] += 1

    def get_usage_stats(self) -> Dict:
        total = len(self.chunk_usage_counts)
        exhausted = sum(1 for c in self.chunk_usage_counts if c >= MAX_CHUNK_USAGE)
        avg_usage = sum(self.chunk_usage_counts) / total if total > 0 else 0

        return {
            "total_chunks": total,
            "exhausted_chunks": exhausted,
            "available_chunks": total - exhausted,
            "avg_usage": avg_usage,
            "exhausted_percent": (exhausted / total * 100) if total > 0 else 0
        }


class FastDatasetProcessor:
    def __init__(self, model: SentenceTransformer, pool_manager: NegativePoolManager):
        self.model = model
        self.pool_manager = pool_manager
        self.pos_embeddings_cache = {}
        self.processed_queries = 0
        self.generated_triplets = 0

    def encode_batch_optimized(self, texts: List[str], desc: str,
                               batch_size: int = None) -> torch.Tensor:
        if not texts:
            dim = self.model.get_sentence_embedding_dimension()
            return torch.empty(0, dim, dtype=DTYPE, device=DEVICE)

        batch_size = batch_size or EMBEDDING_BATCH_SIZE

        with torch.no_grad():
            embeddings = []
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

            for idx, batch_texts in enumerate(batches):
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

                if idx % 20 == 0:
                    aggressive_cleanup()

            result = torch.cat(embeddings, dim=0) if embeddings else torch.empty(0,
                                                                                 self.model.get_sentence_embedding_dimension(),
                                                                                 dtype=DTYPE, device=DEVICE)

        return result

    def mine_negatives_with_usage_limit(self, query_embs: torch.Tensor,
                                        pos_sims: List[float],
                                        target_margin: float,
                                        tolerance: float) -> List[Tuple[str, float, str, int]]:

        if self.pool_manager.global_neg_pool is None:
            logger.error("No global negative pool available!")
            return [("fallback_negative", 0.0, "emergency", -1)] * len(pos_sims)

        available_indices = self.pool_manager.get_available_indices()

        if len(available_indices) == 0:
            logger.warning("All chunks exhausted! Resetting usage counts")
            self.pool_manager.chunk_usage_counts = [0] * len(self.pool_manager.chunk_usage_counts)
            available_indices = self.pool_manager.get_available_indices()

        logger.debug(f"Mining negatives for {len(query_embs)} queries from {len(available_indices)} available chunks")

        neg_pool = self.pool_manager.global_neg_pool[available_indices]
        neg_texts = [self.pool_manager.global_neg_texts[i] for i in available_indices.tolist()]

        results = []
        selected_indices = []
        margin_low = target_margin - tolerance
        margin_high = target_margin + tolerance

        logger.debug("Computing similarity matrix...")
        similarities = util.cos_sim(query_embs, neg_pool)
        logger.debug("Similarity computation complete")

        for i, pos_sim in enumerate(pos_sims):
            query_sims = similarities[i]

            valid_mask = (query_sims >= MIN_NEG_SIM) & (query_sims <= min(pos_sim * 0.88, MAX_POS_NEG_SIM))
            valid_local_indices = torch.where(valid_mask)[0]

            if len(valid_local_indices) == 0:
                fallback_mask = query_sims < pos_sim * 0.90
                fallback_local_indices = torch.where(fallback_mask)[0]
                if len(fallback_local_indices) > 0:
                    best_local_idx = fallback_local_indices[torch.argmax(query_sims[fallback_local_indices])]
                    best_global_idx = available_indices[best_local_idx].item()
                    results.append(
                        (neg_texts[best_local_idx], float(query_sims[best_local_idx]), "fallback", best_global_idx))
                    selected_indices.append(best_global_idx)
                    monitor.increment("fallback_negatives")
                else:
                    best_local_idx = torch.randint(0, len(neg_texts), (1,))[0]
                    best_global_idx = available_indices[best_local_idx].item()
                    results.append((neg_texts[best_local_idx], float(query_sims[best_local_idx]), "random_fallback",
                                    best_global_idx))
                    selected_indices.append(best_global_idx)
                    monitor.increment("random_fallback_negatives")
                continue

            valid_sims = query_sims[valid_local_indices]
            margins = pos_sim - valid_sims

            semi_hard_mask = (margins >= margin_low) & (margins <= margin_high)
            semi_hard_local_indices = valid_local_indices[semi_hard_mask]

            hard_mask = margins > margin_high
            hard_local_indices = valid_local_indices[hard_mask]

            easy_mask = (margins >= MIN_MARGIN) & (margins < margin_low)
            easy_local_indices = valid_local_indices[easy_mask]

            if len(semi_hard_local_indices) > 0:
                semi_hard_margins = margins[semi_hard_mask]
                best_within_semi = torch.argmin(torch.abs(semi_hard_margins - target_margin))
                best_local_idx = semi_hard_local_indices[best_within_semi]
                best_global_idx = available_indices[best_local_idx].item()
                best_sim = float(valid_sims[torch.where(valid_local_indices == best_local_idx)[0][0]])
                text = neg_texts[best_local_idx]
                results.append((text, best_sim, "semi-hard", best_global_idx))
                selected_indices.append(best_global_idx)
                monitor.increment("semi_hard_negatives")
            elif len(hard_local_indices) > 0:
                hard_margins = margins[hard_mask]
                best_within_hard = torch.argmin(hard_margins)
                best_local_idx = hard_local_indices[best_within_hard]
                best_global_idx = available_indices[best_local_idx].item()
                best_sim = float(valid_sims[torch.where(valid_local_indices == best_local_idx)[0][0]])
                text = neg_texts[best_local_idx]
                results.append((text, best_sim, "hard", best_global_idx))
                selected_indices.append(best_global_idx)
                monitor.increment("hard_negatives")
            elif len(easy_local_indices) > 0:
                easy_margins = margins[easy_mask]
                best_within_easy = torch.argmax(easy_margins)
                best_local_idx = easy_local_indices[best_within_easy]
                best_global_idx = available_indices[best_local_idx].item()
                best_sim = float(valid_sims[torch.where(valid_local_indices == best_local_idx)[0][0]])
                text = neg_texts[best_local_idx]
                results.append((text, best_sim, "easy", best_global_idx))
                selected_indices.append(best_global_idx)
                monitor.increment("easy_negatives")
            else:
                best_within_valid = torch.argmax(margins)
                best_local_idx = valid_local_indices[best_within_valid]
                best_global_idx = available_indices[best_local_idx].item()
                best_sim = float(valid_sims[best_within_valid])
                text = neg_texts[best_local_idx]
                results.append((text, best_sim, "valid_fallback", best_global_idx))
                selected_indices.append(best_global_idx)
                monitor.increment("valid_fallback_negatives")

        self.pool_manager.increment_usage(selected_indices)
        logger.debug(f"Mined {len(results)} negatives")

        del similarities
        aggressive_cleanup()

        return results

    def save_checkpoint(self, triplets: List[Dict], checkpoint_num: int):
        checkpoint_path = os.path.join(TRIPLET_CHECKPOINT_DIR, f"checkpoint_{checkpoint_num:05d}.json")
        usage_path = os.path.join(TRIPLET_CHECKPOINT_DIR, f"usage_{checkpoint_num:05d}.pkl")

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(triplets, f, ensure_ascii=False, indent=2)

        save_pkl_atomic({
            'chunk_usage_counts': self.pool_manager.chunk_usage_counts,
            'num_triplets': len(triplets)
        }, usage_path)

        logger.info(f"Checkpoint saved: {len(triplets):,} triplets")

    def load_latest_checkpoint(self) -> Tuple[List[Dict], int]:
        if not os.path.exists(TRIPLET_CHECKPOINT_DIR):
            return [], 0

        checkpoints = [f for f in os.listdir(TRIPLET_CHECKPOINT_DIR) if
                       f.startswith("checkpoint_") and f.endswith(".json")]

        if not checkpoints:
            return [], 0

        latest = max(checkpoints)
        checkpoint_num = int(latest.split("_")[1].split(".")[0])

        checkpoint_path = os.path.join(TRIPLET_CHECKPOINT_DIR, latest)
        usage_path = os.path.join(TRIPLET_CHECKPOINT_DIR, f"usage_{checkpoint_num:05d}.pkl")

        logger.info(f"Loading checkpoint {checkpoint_num}...")

        with open(checkpoint_path, "r", encoding="utf-8") as f:
            triplets = json.load(f)

        if os.path.exists(usage_path):
            usage_data = load_pkl_safe(usage_path, {})
            if 'chunk_usage_counts' in usage_data:
                self.pool_manager.chunk_usage_counts = usage_data['chunk_usage_counts']

        logger.info(f"Loaded {len(triplets):,} triplets from checkpoint")

        return triplets, checkpoint_num

    def process_dataset_batch(self, documents: List[Dict],
                              dataset_idx: int, total_datasets: int) -> List[Dict]:
        logger.info(f"Processing dataset with {len(documents):,} docs for 125K triplets")

        all_triplets, last_checkpoint = self.load_latest_checkpoint()

        if len(all_triplets) >= TARGET_TRIPLETS:
            logger.info(f"Already have {len(all_triplets):,} triplets from checkpoints, done!")
            return all_triplets[:TARGET_TRIPLETS]

        logger.info("Loading document cache...")
        records = []

        oids = [doc["_id"]["$oid"] for doc in documents]

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            cache_futures = {oid: executor.submit(self._load_cache_doc, oid) for oid in oids}
            cache_data = {oid: future.result() for oid, future in cache_futures.items()}

        logger.info("Extracting query-answer pairs...")
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
            logger.error("No valid records found!")
            return []

        logger.info(f"Extracted {len(records):,} query-answer pairs")

        random.shuffle(records)

        del cache_data
        aggressive_cleanup()

        logger.info(f"Starting triplet generation (have {len(all_triplets):,}, need {TARGET_TRIPLETS:,})")

        batch_size = 2000
        checkpoint_counter = last_checkpoint

        for batch_start in range(0, len(records), batch_size):
            if len(all_triplets) >= TARGET_TRIPLETS:
                logger.info(f"Reached target of {TARGET_TRIPLETS:,} triplets!")
                break

            batch_end = min(batch_start + batch_size, len(records))
            batch_records = records[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start:,}-{batch_end:,} ({len(batch_records):,} records)")

            usage_stats = self.pool_manager.get_usage_stats()
            logger.info(
                f"Pool: {usage_stats['available_chunks']:,} available, {usage_stats['exhausted_percent']:.1f}% exhausted")

            logger.info("Encoding queries...")
            query_texts = [f"query: {rec['query']}" for rec in batch_records]
            query_embeddings = self.encode_batch_optimized(query_texts, f"queries", QUERY_BATCH_SIZE)
            logger.info(f"Encoded {len(query_embeddings)} query embeddings")

            target_margin, tolerance, difficulty = self._get_curriculum_params(dataset_idx, total_datasets)

            logger.info(f"Generating triplets (margin={target_margin:.2f}±{tolerance:.2f})...")
            batch_triplets = self._generate_triplets_batch_guaranteed(
                batch_records, query_embeddings, target_margin, tolerance, difficulty
            )

            all_triplets.extend(batch_triplets)
            logger.info(
                f"Generated {len(batch_triplets):,} triplets (total: {len(all_triplets):,}/{TARGET_TRIPLETS:,})")

            del query_embeddings
            del batch_records
            aggressive_cleanup()

            if len(all_triplets) % CHECKPOINT_EVERY < len(batch_triplets) or len(all_triplets) >= TARGET_TRIPLETS:
                checkpoint_counter += 1
                logger.info(f"Saving checkpoint {checkpoint_counter}...")
                self.save_checkpoint(all_triplets, checkpoint_counter)

        if len(all_triplets) > TARGET_TRIPLETS:
            logger.info(f"Sampling best {TARGET_TRIPLETS:,} triplets from {len(all_triplets):,}")

            ideal_margin = 0.20
            all_triplets.sort(key=lambda x: abs(x.get("margin", 0) - ideal_margin))
            all_triplets = all_triplets[:TARGET_TRIPLETS]

            random.shuffle(all_triplets)

        logger.info("Saving final dataset...")
        output_file = os.path.join(DATASETS_DIR, f"dataset_1.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_triplets, f, ensure_ascii=False, indent=2)

        logger.info("Computing statistics...")
        type_counts = defaultdict(int)
        margins = []
        pos_sims = []
        neg_sims = []

        for t in all_triplets:
            type_counts[t.get("type", "unknown")] += 1
            if "margin" in t:
                margins.append(t["margin"])
            if "pos_sim" in t:
                pos_sims.append(t["pos_sim"])
            if "neg_sim" in t:
                neg_sims.append(t["neg_sim"])

        logger.info(f"Dataset Summary:")
        logger.info(f"  Total triplets: {len(all_triplets):,}")
        logger.info(f"  Triplet types: {dict(type_counts)}")

        if margins:
            logger.info(
                f"  Margins - Mean: {np.mean(margins):.3f}, Std: {np.std(margins):.3f}, Median: {np.median(margins):.3f}")
        if pos_sims:
            logger.info(f"  Positive sims - Mean: {np.mean(pos_sims):.3f}, Std: {np.std(pos_sims):.3f}")
        if neg_sims:
            logger.info(f"  Negative sims - Mean: {np.mean(neg_sims):.3f}, Std: {np.std(neg_sims):.3f}")

        final_usage_stats = self.pool_manager.get_usage_stats()
        logger.info(f"Final pool usage: {final_usage_stats}")

        return all_triplets

    def _load_cache_doc(self, oid: str) -> Dict:
        cache_path = os.path.join(CACHE_DIR, f"{oid}.pkl")
        return load_pkl_safe(cache_path, {})

    def _get_curriculum_params(self, dataset_idx: int, total_datasets: int) -> Tuple[float, float, float]:
        margin = 0.20
        tolerance = 0.08
        difficulty = 0.6

        return float(margin), float(tolerance), float(difficulty)

    def _generate_triplets_batch_guaranteed(self, records: List[Dict],
                                            query_embeddings: torch.Tensor,
                                            target_margin: float, tolerance: float,
                                            difficulty: float) -> List[Dict]:
        triplets = []
        batch_size = 128

        total_batches = (len(records) + batch_size - 1) // batch_size
        logger.info(f"Processing {len(records)} records in {total_batches} batches of {batch_size}")

        for batch_idx in range(0, len(records), batch_size):
            batch_records = records[batch_idx:batch_idx + batch_size]
            batch_query_embs = query_embeddings[batch_idx:batch_idx + batch_size]

            logger.debug(f"Finding positives for batch {batch_idx // batch_size + 1}/{total_batches}...")
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

            logger.debug(f"Mining {len(valid_query_embs)} negatives...")
            negative_results = self.mine_negatives_with_usage_limit(
                valid_query_embs, pos_sims, target_margin, tolerance
            )

            for (rec, pos_result), neg_result in zip(valid_records, negative_results):
                pos_text, pos_sim = pos_result
                neg_text, neg_sim, neg_type, neg_idx = neg_result

                triplet = {
                    "doc_id": rec["oid"],
                    "question": rec["query"],
                    "positive": pos_text,
                    "negative": neg_text,
                    "pos_sim": float(pos_sim),
                    "neg_sim": float(neg_sim),
                    "margin": float(pos_sim - neg_sim),
                    "type": neg_type,
                    "pool_chunk_idx": neg_idx
                }
                triplets.append(triplet)
                monitor.increment("triplets_generated")

            if batch_idx % (batch_size * 5) == 0:
                aggressive_cleanup()

        logger.info(f"Generated {len(triplets)} triplets from this batch")
        return triplets

    def _find_positives_batch_guaranteed(self, query_embs: torch.Tensor,
                                         records: List[Dict]) -> List[Optional[Tuple[str, float]]]:
        results = []

        all_pos_texts = []
        record_indices = []
        text_to_record = []

        for i, record in enumerate(records):
            correct_chunks = record["correct_chunks"]
            if not correct_chunks:
                results.append((f"Fallback positive for {record['query'][:50]}", 0.6))
                continue

            start_idx = len(all_pos_texts)
            pos_texts = [f"passage: {chunk}" for chunk in correct_chunks]
            all_pos_texts.extend(pos_texts)

            for j in range(len(pos_texts)):
                record_indices.append(i)
                text_to_record.append((i, correct_chunks[j]))

        if not all_pos_texts:
            return results

        pos_embs = self.encode_batch_optimized(all_pos_texts, "positives", EMBEDDING_BATCH_SIZE)

        emb_idx = 0
        for i, record in enumerate(records):
            if i < len(results):
                continue

            correct_chunks = record["correct_chunks"]
            num_chunks = len(correct_chunks)

            chunk_embs = pos_embs[emb_idx:emb_idx + num_chunks]
            emb_idx += num_chunks

            if chunk_embs.numel() == 0:
                results.append((correct_chunks[0], 0.5))
                continue

            query_emb = query_embs[i:i + 1]
            similarities = util.cos_sim(query_emb, chunk_embs)[0]

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


def main(args):
    logger.info("Starting Optimized Triplet Mining for 125K Triplets with Sharded Pool")
    logger.info(f"   Device: {DEVICE} ({DTYPE})")
    logger.info(f"   Target: {TARGET_TRIPLETS:,} triplets in single dataset")
    logger.info(f"   Max chunk usage: {MAX_CHUNK_USAGE}")
    logger.info(f"   Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    logger.info(f"   Checkpoint every: {CHECKPOINT_EVERY:,} triplets")
    logger.info(f"   Similarity batch size: {SIMILARITY_BATCH_SIZE:,}")

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    ensure_dirs()
    monitor.start_stage("initialization")

    model = load_model_optimized(args.model)

    logger.info("Loading tokenizer for chunking...")
    tokenizer = AutoTokenizer.from_pretrained("PORTULAN/albertina-1b5-portuguese-ptpt-encoder")

    logger.info("Initializing negative pool manager...")
    pool_manager = NegativePoolManager(POOL_PATH, model, tokenizer)

    processor = FastDatasetProcessor(model, pool_manager)

    monitor.end_stage("initialization")

    logger.info("Loading all documents...")
    monitor.start_stage("dataset_loading")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        all_docs = json.load(f)

    logger.info(f"Loaded {len(all_docs):,} documents")
    monitor.end_stage("dataset_loading")

    logger.info("Processing all documents to generate 125K triplets...")
    triplets = processor.process_dataset_batch(all_docs, 1, 1)

    logger.info(f"Successfully generated {len(triplets):,} triplets!")
    logger.info(f"Saved to {os.path.join(DATASETS_DIR, 'dataset_1.json')}")

    monitor.log_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Triplet Mining with Sharded Pool")
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-base",
                        help="SentenceTransformer model name")
    parser.add_argument("--target-triplets", type=int, default=TARGET_TRIPLETS,
                        help="Target number of triplets")
    parser.add_argument("--embedding-batch-size", type=int, default=EMBEDDING_BATCH_SIZE,
                        help="Batch size for embedding generation")
    parser.add_argument("--query-batch-size", type=int, default=QUERY_BATCH_SIZE,
                        help="Batch size for query processing")
    parser.add_argument("--neg-candidates", type=int, default=NEG_CANDIDATES,
                        help="Number of negative candidates to sample")
    parser.add_argument("--max-chunk-usage", type=int, default=MAX_CHUNK_USAGE,
                        help="Maximum times a chunk can be used as negative")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help="Size of pool document chunks")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP,
                        help="Overlap between chunks")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                        help="Save checkpoint every N triplets")
    parser.add_argument("--similarity-batch", type=int, default=SIMILARITY_BATCH_SIZE,
                        help="Batch size for similarity computation")
    parser.add_argument("--memory-aggressive", action="store_true",
                        help="Enable aggressive memory management")
    parser.add_argument("--rebuild-pool", action="store_true",
                        help="Force rebuild of pool cache")
    parser.add_argument("--clear-checkpoints", action="store_true",
                        help="Clear existing checkpoints and start fresh")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger(__name__).setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")

    TARGET_TRIPLETS = args.target_triplets
    EMBEDDING_BATCH_SIZE = args.embedding_batch_size
    QUERY_BATCH_SIZE = args.query_batch_size
    NEG_CANDIDATES = args.neg_candidates
    MAX_CHUNK_USAGE = args.max_chunk_usage
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    CHECKPOINT_EVERY = args.checkpoint_every
    SIMILARITY_BATCH_SIZE = args.similarity_batch

    if args.clear_checkpoints:
        import shutil

        if os.path.exists(TRIPLET_CHECKPOINT_DIR):
            shutil.rmtree(TRIPLET_CHECKPOINT_DIR)
            logger.info(f"Cleared checkpoint directory")
        ensure_dirs()

    if args.rebuild_pool:
        cache_path = os.path.join(POOL_CACHE_DIR, "pool_embeddings.pt")
        metadata_path = os.path.join(POOL_CACHE_DIR, "pool_metadata.pkl")
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info(f"Removed {cache_path}")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            logger.info(f"Removed {metadata_path}")

    if args.memory_aggressive:
        EMBEDDING_BATCH_SIZE = max(32, EMBEDDING_BATCH_SIZE // 2)
        QUERY_BATCH_SIZE = max(32, QUERY_BATCH_SIZE // 2)
        SIMILARITY_BATCH_SIZE = max(512, SIMILARITY_BATCH_SIZE // 2)
        logger.info("Aggressive memory mode enabled - reduced batch sizes")

    try:
        main(args)
        logger.info("Process completed successfully!")
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
