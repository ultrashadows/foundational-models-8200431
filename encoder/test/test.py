"""Retrieval evaluation (recall@k and MRR) of an encoder on question-to-document search.

Every generated question (``results[].pergunta``) is a query and its source decision is the only
relevant document. Long documents are split into overlapping chunks, embedded with the chosen
pooling, and averaged into one vector per document.
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
ENCODER_DIR = HERE.parent


class IREvaluator:
    def __init__(self, model_path: str, device: str = "cuda", batch_size: int = 128, chunk_size: int = 512,
                 overlap: int = 128, cache_dir: str = "./cache", sim_block_docs: int = 2048,
                 max_chunks_per_doc: int = 400, min_overlap: int = 32, pooling: str = "mean"):
        self.device = device
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.cache_dir = cache_dir
        self.sim_block_docs = sim_block_docs
        self.max_chunks_per_doc = max_chunks_per_doc
        self.min_overlap = min_overlap
        self.pooling = pooling  # "mean", "cls", or "max"
        self.model = None
        self.tokenizer = None
        self.hidden_size = None
        self.load_model(model_path)
        # Document embeddings depend on the model and on how documents are chunked and pooled,
        # so the cache is split per configuration. A single shared folder would silently reuse
        # one model's embeddings when evaluating another.
        config_key = f"{model_path}|{pooling}|{chunk_size}|{overlap}|{max_chunks_per_doc}|{min_overlap}"
        config_dir = f"{os.path.basename(os.path.normpath(model_path))}-{hashlib.sha1(config_key.encode()).hexdigest()[:10]}"
        self.embeddings_cache_dir = os.path.join(cache_dir, "embeddings", config_dir)
        os.makedirs(self.embeddings_cache_dir, exist_ok=True)

    def load_model(self, model_path: str):
        logger.info(f"Loading model from {model_path}")

        # Check if model_path is a local directory
        if os.path.isdir(model_path):
            logger.info(f"Loading fine-tuned model from local directory: {model_path}")

            # Verify required files exist
            required_files = ["config.json", "tokenizer.json"]
            safetensors_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]

            missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
            if missing_files:
                raise FileNotFoundError(f"Missing required files: {missing_files}")

            if not safetensors_files:
                raise FileNotFoundError("No .safetensors files found in model directory")

            logger.info(f"Found safetensors files: {safetensors_files}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                                       local_files_only=os.path.isdir(model_path))

        if self.tokenizer.pad_token is None:
            if getattr(self.tokenizer, "eos_token", None) is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with safetensors support
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=os.path.isdir(model_path)
        ).to(self.device)

        self.model.eval()
        self.hidden_size = self.model.config.hidden_size
        logger.info(f"Model loaded successfully (hidden_size={self.hidden_size})")

    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling with proper masking"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_cache_filename(self, doc_oid: str) -> str:
        return os.path.join(self.embeddings_cache_dir, f"doc_{doc_oid}.npy")

    def save_document_embedding(self, doc_oid: str, embedding: torch.Tensor):
        cache_file = self.get_cache_filename(doc_oid)
        np.save(cache_file, embedding.cpu().numpy().astype(np.float16))

    def load_document_embedding(self, doc_oid: str) -> torch.Tensor:
        cache_file = self.get_cache_filename(doc_oid)
        if os.path.exists(cache_file):
            return torch.from_numpy(np.load(cache_file)).float()
        return None

    @torch.inference_mode()
    def encode_batch(self, texts: List[str] = None,
                     input_ids: torch.Tensor = None,
                     attention_mask: torch.Tensor = None,
                     normalize: bool = True) -> torch.Tensor:
        assert (texts is not None) or (input_ids is not None and attention_mask is not None)

        if texts is not None:
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.chunk_size,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)

        bs = max(1, min(self.batch_size, input_ids.size(0)))
        outs = []
        start = 0

        while start < input_ids.size(0):
            cur_bs = min(bs, input_ids.size(0) - start)
            while True:
                try:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(
                            input_ids[start:start + cur_bs],
                            attention_mask=attention_mask[start:start + cur_bs]
                        )

                    # Get token embeddings
                    token_embeddings = outputs.last_hidden_state

                    # Apply pooling strategy
                    if self.pooling == "cls":
                        embs = token_embeddings[:, 0, :]
                    elif self.pooling == "mean":
                        embs = self.mean_pooling(
                            token_embeddings,
                            attention_mask[start:start + cur_bs]
                        )
                    elif self.pooling == "max":
                        embs = torch.max(token_embeddings, dim=1).values
                    else:
                        raise ValueError(f"Unknown pooling: {self.pooling}")

                    # Normalize embeddings
                    if normalize:
                        embs = F.normalize(embs, p=2, dim=1)

                    outs.append(embs.detach().cpu())
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if cur_bs == 1:
                        raise
                    cur_bs = max(1, cur_bs // 2)
            start += cur_bs

        return torch.cat(outs, dim=0)

    def _token_count(self, text: str) -> int:
        enc = self.tokenizer(text, add_special_tokens=False, return_length=True)
        if "length" in enc and isinstance(enc["length"], list):
            return int(enc["length"][0])
        if "length" in enc and isinstance(enc["length"], int):
            return int(enc["length"])
        if "input_ids" in enc and isinstance(enc["input_ids"], list):
            return len(enc["input_ids"])
        return 0

    def _adaptive_overflow(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        L = self._token_count(text)
        if L <= self.chunk_size:
            enc = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.chunk_size,
                return_tensors="pt"
            )
            return enc["input_ids"], enc["attention_mask"]

        raw_stride = self.chunk_size - self.overlap if self.chunk_size > self.overlap else self.chunk_size
        target_stride = max(1, (L - self.chunk_size) // max(1, self.max_chunks_per_doc - 1))
        stride = max(self.min_overlap, min(raw_stride, target_stride))

        enc = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.chunk_size,
            stride=stride,
            return_overflowing_tokens=True,
            return_tensors="pt"
        )
        return enc["input_ids"], enc["attention_mask"]

    @torch.inference_mode()
    def encode_long_document(self, text: str, use_weighted_pooling: bool = False) -> torch.Tensor:
        input_ids, attention_mask = self._adaptive_overflow(text)

        if input_ids.size(0) == 0:
            empty = self.encode_batch(texts=[""])
            return empty.squeeze(0)

        outs = []
        bs = max(1, min(self.batch_size, input_ids.size(0)))
        for start in tqdm(range(0, input_ids.size(0), bs), desc="Doc chunks", leave=False):
            end = min(start + bs, input_ids.size(0))
            batch_ids = input_ids[start:end]
            batch_mask = attention_mask[start:end]
            embs = self.encode_batch(input_ids=batch_ids, attention_mask=batch_mask).cpu()
            outs.append(embs)

        embs_all = torch.cat(outs, dim=0)

        if use_weighted_pooling:
            # Give more weight to earlier chunks (often contain key info)
            num_chunks = embs_all.size(0)
            weights = torch.linspace(1.0, 0.5, num_chunks).unsqueeze(1)
            doc_embed = (embs_all * weights).sum(dim=0) / weights.sum()
        else:
            # Standard mean pooling
            doc_embed = embs_all.mean(dim=0)

        # Re-normalize after averaging
        doc_embed = F.normalize(doc_embed.unsqueeze(0), p=2, dim=1).squeeze(0)

        return doc_embed

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        all_embeddings = []
        for i in tqdm(range(0, len(queries), self.batch_size), desc="Encoding queries"):
            batch = queries[i:i + self.batch_size]
            embeds = self.encode_batch(texts=batch)
            all_embeddings.append(embeds.cpu())
        return torch.cat(all_embeddings, dim=0)

    def encode_documents(self, documents: List[str], doc_oids: List[str] = None) -> torch.Tensor:
        cached_pairs = []
        to_encode_idx = []

        for idx, doc in enumerate(documents):
            doc_oid = doc_oids[idx] if doc_oids and idx < len(doc_oids) else None
            if doc_oid:
                cached_emb = self.load_document_embedding(doc_oid)
                if cached_emb is not None:
                    cached_pairs.append((idx, cached_emb))
                    continue
            to_encode_idx.append(idx)

        all_embeddings = [None] * len(documents)
        for idx, emb in cached_pairs:
            all_embeddings[idx] = emb

        for idx in tqdm(to_encode_idx, desc="Encoding documents"):
            doc_oid = doc_oids[idx] if doc_oids and idx < len(doc_oids) else None
            emb = self.encode_long_document(documents[idx])
            all_embeddings[idx] = emb
            if doc_oid:
                self.save_document_embedding(doc_oid, emb)

        logger.info(f"Loaded {len(cached_pairs)} from cache, encoded {len(to_encode_idx)} new documents")
        return torch.stack(all_embeddings)

    def prepare_evaluation_data(self, eval_dataset_path: str, cache_dir: str = None) -> Tuple[
        List[str], List[str], List[str], Dict[str, str], List[str]]:
        logger.info(f"Loading test dataset from {eval_dataset_path}")
        with open(eval_dataset_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        logger.info(f"Loaded {len(eval_data)} test documents")

        queries = []
        correct_doc_ids = []
        doc_id_to_text = {}
        all_doc_texts = []
        all_doc_oids = []
        seen_texts = set()

        for doc in tqdm(eval_data, desc="Processing test documents"):
            doc_id = doc["_id"]["$oid"]
            doc_text = ""

            if cache_dir:
                cache_path = os.path.join(cache_dir, f"{doc_id}.pkl")
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, 'rb') as f:
                            cache_data = pickle.load(f)
                        if "correct_doc_text" in cache_data and cache_data["correct_doc_text"]:
                            doc_text = cache_data["correct_doc_text"]
                        elif "correct_chunks" in cache_data and cache_data["correct_chunks"]:
                            doc_text = " ".join(cache_data["correct_chunks"])
                    except Exception as e:
                        logger.warning(f"Failed to load cache for {doc_id}: {e}")

            if not doc_text and "Texto Integral" in doc:
                doc_text = doc["Texto Integral"]

            if not doc_text:
                logger.warning(f"No text found for document {doc_id}")
                continue

            doc_id_to_text[doc_id] = doc_text
            if doc_text not in seen_texts:
                seen_texts.add(doc_text)
                all_doc_texts.append(doc_text)
                all_doc_oids.append(doc_id)

            for result in doc.get("results", []):
                query = result.get("pergunta", "").strip()
                if len(query) > 5:
                    queries.append(query)
                    correct_doc_ids.append(doc_id)

        logger.info(f"Extracted {len(queries)} queries from {len(set(correct_doc_ids))} unique documents")
        logger.info(f"Total unique document texts: {len(all_doc_texts)}")
        return queries, all_doc_texts, correct_doc_ids, doc_id_to_text, all_doc_oids

    @torch.inference_mode()
    def _query_topk_and_rank(self, query_vec: torch.Tensor, doc_embs_cpu: torch.Tensor,
                             k_max: int, correct_idx: int) -> Tuple[List[int], List[float], float, int]:
        q_dev = query_vec.to(self.device, non_blocking=True)

        # Embeddings are normalised, so the dot product is the cosine similarity. Document blocks
        # are moved to the device one at a time to bound memory. The rank is computed from the
        # same scores as the top-k, so the correct document is never compared against a score
        # computed at a different precision.
        blocks = []
        for start in range(0, doc_embs_cpu.size(0), self.sim_block_docs):
            block = doc_embs_cpu[start:start + self.sim_block_docs].to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                blocks.append(torch.matmul(q_dev.unsqueeze(0), block.T).squeeze(0).float())
        sims = torch.cat(blocks)

        correct_sim = float(sims[correct_idx])
        top_scores, top_indices = torch.topk(sims, k=min(k_max, sims.numel()))
        rank = int((sims > sims[correct_idx]).sum().item()) + 1
        return top_indices.cpu().tolist(), [float(x) for x in top_scores.cpu()], correct_sim, rank

    def evaluate_retrieval(self, queries: List[str], documents: List[str],
                           correct_doc_ids: List[str], doc_id_to_text: Dict[str, str],
                           doc_oids: List[str] = None, k_values=None) -> Dict[str, float]:
        if k_values is None:
            k_values = [1, 3, 5, 10, 20]

        logger.info(f"Evaluating retrieval on {len(queries)} queries against {len(documents)} documents")
        start_time = time.time()

        query_embeddings = self.encode_queries(queries)
        doc_embeddings = self.encode_documents(documents, doc_oids)

        encoding_time = time.time() - start_time
        logger.info(f"Encoding completed in {encoding_time:.2f}s")

        doc_text_to_idx = {doc_text: idx for idx, doc_text in enumerate(documents)}
        k_max = max(k_values)

        hits_by_k = {k: 0 for k in k_values}
        reciprocal_ranks = []

        for i in tqdm(range(len(queries)), desc="Scoring queries"):
            q_vec = query_embeddings[i]
            correct_doc_text = doc_id_to_text.get(correct_doc_ids[i], "")
            if correct_doc_text not in doc_text_to_idx:
                continue

            correct_idx = doc_text_to_idx[correct_doc_text]
            top_idx, _, _, rank = self._query_topk_and_rank(q_vec, doc_embeddings, k_max, correct_idx)

            for k in k_values:
                if correct_idx in set(top_idx[:k]):
                    hits_by_k[k] += 1

            reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)

        results = {}
        total_q = len(queries) if len(queries) > 0 else 1
        for k in k_values:
            results[f"recall@{k}"] = hits_by_k[k] / total_q
        results["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

        self._cache_eval_state = {
            "query_embeddings": query_embeddings,
            "doc_embeddings": doc_embeddings,
            "documents": documents,
            "doc_text_to_idx": doc_text_to_idx,
            "doc_id_to_text": doc_id_to_text,
            "k_max": k_max
        }

        return results

    def analyze_failure_cases(self, queries: List[str], correct_doc_ids: List[str], num_cases: int = 10) -> List[Dict]:
        logger.info(f"Analyzing {num_cases} failure cases")
        if not hasattr(self, "_cache_eval_state"):
            logger.error("Must run evaluate_retrieval before analyze_failure_cases")
            return []

        S = self._cache_eval_state
        q_embs = S["query_embeddings"]
        d_embs = S["doc_embeddings"]
        documents = S["documents"]
        doc_text_to_idx = S["doc_text_to_idx"]
        doc_id_to_text = S["doc_id_to_text"]
        k = 5

        failure_cases = []
        for i in range(min(len(queries), q_embs.size(0))):
            query = queries[i]
            correct_doc_id = correct_doc_ids[i]
            correct_doc_text = doc_id_to_text.get(correct_doc_id, "")
            if correct_doc_text not in doc_text_to_idx:
                continue

            correct_idx = doc_text_to_idx[correct_doc_text]
            top_idx, top_scores, correct_sim, rank = self._query_topk_and_rank(q_embs[i], d_embs, k, correct_idx)
            top_docs = [documents[j] for j in top_idx[:k]]

            if correct_idx not in set(top_idx[:k]):
                case = {
                    "query": query,
                    "correct_doc_text": correct_doc_text[:200] + "..." if len(
                        correct_doc_text) > 200 else correct_doc_text,
                    "correct_similarity": float(correct_sim),
                    "correct_rank": int(rank),
                    "top_5_retrieved": [
                        {
                            "text": d[:100] + "..." if len(d) > 100 else d,
                            "similarity": float(s)
                        } for d, s in zip(top_docs, top_scores[:k])
                    ]
                }
                failure_cases.append(case)
                if len(failure_cases) >= num_cases:
                    break

        return failure_cases


def compare_models(base_results: Dict, finetuned_results: Dict):
    """Compare base model vs fine-tuned model performance"""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<15} {'Base Model':<15} {'Fine-tuned':<15} {'Improvement':<20}")
    print("-" * 70)

    for metric in ["recall@1", "recall@3", "recall@5", "recall@10", "recall@20", "mrr"]:
        base_score = base_results.get(metric, 0.0)
        ft_score = finetuned_results.get(metric, 0.0)

        if base_score > 0:
            improvement = ((ft_score - base_score) / base_score) * 100
            improvement_str = f"{improvement:+.1f}%"
        else:
            improvement_str = "N/A"

        print(f"{metric:<15} {base_score:<15.4f} {ft_score:<15.4f} {improvement_str:<20}")
    print("=" * 70)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(ENCODER_DIR / "models" / "finetuned-albertina-1b5-100k-triplets"),
                        help="model directory or Hugging Face id to evaluate")
    parser.add_argument("--eval-data", type=Path, default=HERE / "evaluation" / "clean.json",
                        help="evaluation documents (MongoDB extended JSON with results)")
    parser.add_argument("--cache-dir", type=Path, default=ENCODER_DIR / "cache",
                        help="document cache from cache_processor, also used for embedding caches")
    parser.add_argument("--output", type=Path, default=HERE / "finetuned_evaluation_results.json")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Evaluation requires a CUDA GPU. No GPU was found.")
    if not args.eval_data.is_file():
        raise SystemExit(f"Evaluation data not found at {args.eval_data}. Pass --eval-data.")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_path = args.model
    eval_dataset_path = str(args.eval_data)
    cache_dir = str(args.cache_dir)

    logger.info("Initializing model evaluator...")
    evaluator = IREvaluator(
        model_path,
        device="cuda",
        batch_size=128,
        chunk_size=512,
        overlap=128,
        cache_dir=cache_dir,
        sim_block_docs=2048,
        max_chunks_per_doc=400,
        min_overlap=32,
        pooling="mean"
    )

    result = evaluator.prepare_evaluation_data(eval_dataset_path, cache_dir)
    queries, documents, correct_doc_ids, doc_id_to_text, doc_oids = result

    logger.info("Evaluating fine-tuned model...")
    results = evaluator.evaluate_retrieval(queries, documents, correct_doc_ids, doc_id_to_text, doc_oids)

    print("\n" + "=" * 50)
    print("FINE-TUNED MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"{'Metric':<15} {'Score':<12}")
    print("-" * 30)
    for metric in ["recall@1", "recall@3", "recall@5", "recall@10", "recall@20", "mrr"]:
        score = results.get(metric, 0.0)
        print(f"{metric:<15} {score:<12.4f}")

    # Reference scores for the base model, hard-coded in the original script. The run that
    # produced them is not recorded in the repository.
    base_results = {
        "recall@1": 0.0313,
        "recall@3": 0.0660,
        "recall@5": 0.0907,
        "recall@10": 0.1487,
        "recall@20": 0.2253,
        "mrr": 0.0650
    }

    compare_models(base_results, results)

    if results.get("recall@5", 0) < 0.5:
        logger.info("Analyzing failure cases...")
        failure_cases = evaluator.analyze_failure_cases(queries, correct_doc_ids, num_cases=5)
        print("\n" + "=" * 50)
        print("FAILURE CASE ANALYSIS")
        print("=" * 50)
        for i, case in enumerate(failure_cases):
            print(f"\nFailure Case {i + 1}:")
            print(f"Query: {case['query']}")
            print(f"Correct doc similarity: {case['correct_similarity']:.4f} (rank: {case['correct_rank']})")
            print(f"Correct doc text: {case['correct_doc_text']}")
            print("Top 5 retrieved:")
            for j, retrieved in enumerate(case['top_5_retrieved']):
                print(f"  {j + 1}. (sim: {retrieved['similarity']:.4f}) {retrieved['text']}")

    results_summary = {
        "model_path": model_path,
        "results": results,
        "base_model_comparison": base_results,
        "evaluation_info": {
            "num_queries": len(queries),
            "num_documents": len(documents),
            "unique_docs": len(set(correct_doc_ids))
        }
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Evaluation complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
