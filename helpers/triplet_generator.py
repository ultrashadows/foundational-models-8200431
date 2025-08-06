import json
import os
import pickle
import time
from tqdm import tqdm
from typing import List, Dict
import torch
from sentence_transformers import SentenceTransformer, util

DATASET_PATH = "../encoder/dataset.json"
CACHE_DIR = "../encoder/cache"
DATASETS_DIR = "../encoder/datasets"
BATCH_SIZE = 1024
FIRST_BATCH_SIZE = 2000
NEXT_BATCH_SIZE = 1600
OVERLAP_PERCENT = 0.2
HARD_NEGATIVE_K = 5
BASE_MARGIN = 0.2
TARGET_TRIPLETS = 8000
MIN_MARGIN = 0.05
MIN_SIM_POS = 0.3

os.makedirs(DATASETS_DIR, exist_ok=True)
model = SentenceTransformer('intfloat/multilingual-e5-base', device="cuda")
model.half()


def load_dataset() -> List[Dict]:
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_cache(oid: str) -> Dict:
    path = os.path.join(CACHE_DIR, f"{oid}.pkl")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"⚠️ Skipping corrupted cache file {oid}: {e}")
        return {}


def precompute_chunk_embeddings(all_chunks: List[str], desc="Embedding chunks") -> torch.Tensor:
    if not all_chunks:
        return torch.empty(0, model.get_sentence_embedding_dimension())
    unique_chunks = list(dict.fromkeys(all_chunks))
    print(f"  • {len(all_chunks):,} total chunks, {len(unique_chunks):,} unique chunks to embed.")
    chunk_texts = [f"passage: {chunk}" for chunk in unique_chunks]
    start_time = time.time()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(chunk_texts), BATCH_SIZE), desc=desc):
            batch = chunk_texts[i:i + BATCH_SIZE]
            batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
            embeddings.append(batch_embeddings)
    embeddings = torch.cat(embeddings, dim=0)
    elapsed = time.time() - start_time
    print(f"  • Finished embedding chunks in {elapsed:.2f} sec ({len(unique_chunks)} embeddings).")
    chunk_to_idx = {chunk: idx for idx, chunk in enumerate(unique_chunks)}
    ordered_embeddings = torch.stack([embeddings[chunk_to_idx[c]] for c in all_chunks])
    return ordered_embeddings


def find_best_and_worst_chunks_vectorized(
    query_embedding,
    correct_chunks,
    wrong_chunks,
    correct_embeddings,
    wrong_embeddings,
    hard_negative_k=5,
    base_margin=0.2,
    dataset_idx=1,
    fallback=False,
    top_k_positive=3
):
    if len(correct_chunks) == 0 or len(wrong_chunks) == 0:
        return None, None
    min_margin = MIN_MARGIN * (0.5 if fallback else 1.0)
    min_sim_pos = MIN_SIM_POS * (0.5 if fallback else 1.0)
    target_margin = BASE_MARGIN

    correct_scores = util.cos_sim(query_embedding, correct_embeddings)[0]
    wrong_scores = util.cos_sim(query_embedding, wrong_embeddings)[0]

    k = min(top_k_positive, len(correct_scores))
    _, top_pos_indices = torch.topk(correct_scores, k)
    selected_pos_idx = top_pos_indices[0].item()
    positive_sim = correct_scores[selected_pos_idx].item()
    positive = correct_chunks[selected_pos_idx]
    positive_emb = correct_embeddings[selected_pos_idx]

    mask = wrong_scores < positive_sim
    candidate_neg_indices = torch.where(mask)[0]
    if len(candidate_neg_indices) == 0:
        return None, None

    margins = positive_sim - wrong_scores[candidate_neg_indices]
    abs_diff = torch.abs(margins - target_margin)
    best_idx = candidate_neg_indices[torch.argmin(abs_diff)]

    negative = wrong_chunks[best_idx]
    negative_emb = wrong_embeddings[best_idx]

    query_emb = query_embedding.squeeze()
    sim_q_p = torch.nn.functional.cosine_similarity(query_emb, positive_emb, dim=0).item()
    sim_q_n = torch.nn.functional.cosine_similarity(query_emb, negative_emb, dim=0).item()
    sim_p_n = torch.nn.functional.cosine_similarity(positive_emb, negative_emb, dim=0).item()

    if sim_q_p < min_sim_pos or sim_q_p - sim_q_n < min_margin:
        return None, None
    if sim_p_n > 0.95:
        return None, None

    return positive, negative


def get_last_dataset_index() -> int:
    files = os.listdir(DATASETS_DIR)
    indices = [int(f.split('_')[1].split('.')[0]) for f in files if f.startswith("dataset_") and f.endswith(".json")]
    return max(indices) if indices else 0


def batched_encode(texts, desc):
    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
            batch = texts[i:i+BATCH_SIZE]
            batch_embs = model.encode(batch, convert_to_tensor=True, show_progress_bar=False).cpu()
            all_embs.append(batch_embs)
    return torch.cat(all_embs, dim=0)


def process_batch(batch_objects: List[Dict], prev_triplets: List[Dict], dataset_idx: int):
    all_correct_chunks = []
    all_wrong_chunks = []
    object_data = []
    print(f"[# {dataset_idx}] Collecting data from {len(batch_objects)} docs...")
    for obj in tqdm(batch_objects, desc=f"[# {dataset_idx}] Collecting data"):
        oid = obj["_id"]["$oid"]
        try:
            cache = load_cache(oid)
        except FileNotFoundError:
            continue
        correct_chunks = cache.get("correct_chunks", [])
        wrong_chunks = cache.get("wrong_chunks", [])
        if not correct_chunks and cache.get("correct_doc_text"):
            correct_chunks = [cache["correct_doc_text"]]
        if not wrong_chunks and cache.get("wrong_doc_text"):
            wrong_chunks = [cache["wrong_doc_text"]]
        if not correct_chunks or not wrong_chunks:
            continue
        all_correct_chunks.extend(correct_chunks)
        all_wrong_chunks.extend(wrong_chunks)
        for result in obj.get("results", []):
            query = result["pergunta"]
            object_data.append({
                "oid": oid,
                "pergunta": query,
                "query": query,
                "correct_chunks": correct_chunks,
                "wrong_chunks": wrong_chunks
            })

    if not object_data:
        print(f"[# {dataset_idx}] ❌ No valid objects to process.")
        return []

    print(f"[# {dataset_idx}] Precomputing embeddings for ALL chunks...")
    all_unique_chunks = list(dict.fromkeys(all_correct_chunks + all_wrong_chunks))
    all_embeddings = precompute_chunk_embeddings(all_unique_chunks, desc=f"[# {dataset_idx}] All chunks")
    chunk_to_idx = {chunk: idx for idx, chunk in enumerate(all_unique_chunks)}

    print(f"[# {dataset_idx}] Embedding {len(object_data):,} queries...")
    start_q_time = time.time()
    query_texts = [f"query: {q['query']}" for q in object_data]
    query_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(query_texts), BATCH_SIZE), desc=f"[# {dataset_idx}] Queries"):
            batch = query_texts[i:i + BATCH_SIZE]
            batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
            query_embeddings.append(batch_embeddings)
    query_embeddings = torch.cat(query_embeddings, dim=0)
    print(f"  • Finished embedding queries in {time.time() - start_q_time:.2f} sec.")

    print(f"[# {dataset_idx}] Generating triplets...")
    triplets = []
    for i, data in enumerate(object_data):
        if len(triplets) >= TARGET_TRIPLETS:
            break
        query_embedding = query_embeddings[i:i + 1]
        correct_indices = [chunk_to_idx[c] for c in data["correct_chunks"] if c in chunk_to_idx]
        wrong_indices = [chunk_to_idx[c] for c in data["wrong_chunks"] if c in chunk_to_idx]
        if not correct_indices or not wrong_indices:
            continue
        obj_correct_embeddings = all_embeddings[correct_indices]
        obj_wrong_embeddings = all_embeddings[wrong_indices]
        positive, negative = find_best_and_worst_chunks_vectorized(
            query_embedding, data["correct_chunks"], data["wrong_chunks"],
            obj_correct_embeddings, obj_wrong_embeddings, dataset_idx=dataset_idx, fallback=False
        )
        if positive and negative:
            triplets.append({
                "doc_id": data["oid"],
                "question": data["pergunta"],
                "positive": positive,
                "negative": negative
            })

    if len(triplets) < TARGET_TRIPLETS:
        needed = TARGET_TRIPLETS - len(triplets)
        print(f"⚠️ Falling back to easier negatives for {needed} triplets...")
        start_fb_time = time.time()
        processed_fb = 0
        for i, data in enumerate(object_data):
            if len(triplets) >= TARGET_TRIPLETS:
                break
            query_embedding = query_embeddings[i:i + 1]
            correct_indices = [chunk_to_idx[c] for c in data["correct_chunks"] if c in chunk_to_idx]
            wrong_indices = [chunk_to_idx[c] for c in data["wrong_chunks"] if c in chunk_to_idx]
            if not correct_indices or not wrong_indices:
                continue
            obj_correct_embeddings = all_embeddings[correct_indices]
            obj_wrong_embeddings = all_embeddings[wrong_indices]
            positive, negative = find_best_and_worst_chunks_vectorized(
                query_embedding, data["correct_chunks"], data["wrong_chunks"],
                obj_correct_embeddings, obj_wrong_embeddings, dataset_idx=dataset_idx, fallback=True
            )
            processed_fb += 1
            if processed_fb % 100 == 0:
                print(f"    • Fallback processed {processed_fb} items...")
            if positive and negative:
                triplets.append({
                    "doc_id": data["oid"],
                    "question": data["pergunta"],
                    "positive": positive,
                    "negative": negative
                })
        print(f"  • Fallback completed in {time.time() - start_fb_time:.2f} sec, processed {processed_fb} candidates.")

    if prev_triplets:
        overlap_count = int(OVERLAP_PERCENT * len(prev_triplets))
        if overlap_count > 0:
            print(f"[# {dataset_idx}] Selecting {overlap_count} hardest negatives from previous dataset...")
            start_ov_time = time.time()
            triplet_texts = [
                (f"query: {t['question']}", f"passage: {t['positive']}", f"passage: {t['negative']}")
                for t in prev_triplets
            ]
            anchor_embs = batched_encode([t[0] for t in triplet_texts], desc=f"[# {dataset_idx}] Overlap queries")
            pos_embs = batched_encode([t[1] for t in triplet_texts], desc=f"[# {dataset_idx}] Overlap positives")
            neg_embs = batched_encode([t[2] for t in triplet_texts], desc=f"[# {dataset_idx}] Overlap negatives")

            neg_sims = util.cos_sim(anchor_embs, neg_embs).diagonal()
            pos_sims = util.cos_sim(anchor_embs, pos_embs).diagonal()
            hardest_mask = neg_sims < pos_sims
            sorted_indices = torch.argsort(neg_sims[hardest_mask], descending=True)
            hardest_indices = torch.arange(len(pos_sims))[hardest_mask][sorted_indices][:overlap_count]
            hardest_indices = hardest_indices.cpu().tolist()
            triplets += [prev_triplets[i] for i in hardest_indices]
            print(f"  • Overlap selection completed in {time.time() - start_ov_time:.2f} sec.")

    output_file = os.path.join(DATASETS_DIR, f"dataset_{dataset_idx}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
    print(f"[# {dataset_idx}] Created {len(triplets)} triplets. Saved to {output_file}")
    return triplets


def main():
    print("Loading full dataset...")
    all_data = load_dataset()
    total_docs = len(all_data)
    print(f"Loaded {total_docs:,} documents.")
    last_idx = get_last_dataset_index()
    dataset_idx = last_idx + 1
    if last_idx == 0:
        current_index = 0
    else:
        current_index = FIRST_BATCH_SIZE + (dataset_idx - 2) * NEXT_BATCH_SIZE
    prev_triplets = []
    if dataset_idx > 1:
        prev_file = os.path.join(DATASETS_DIR, f"dataset_{dataset_idx - 1}.json")
        if os.path.exists(prev_file):
            with open(prev_file, "r", encoding="utf-8") as f:
                prev_triplets = json.load(f)
    while current_index < total_docs:
        batch_size = FIRST_BATCH_SIZE if dataset_idx == 1 else NEXT_BATCH_SIZE
        batch_objects = all_data[current_index:current_index + batch_size]
        print(f"\n[# {dataset_idx}] Processing docs {current_index:,} → {current_index + len(batch_objects):,}")
        if not batch_objects:
            break
        triplets = process_batch(batch_objects, prev_triplets, dataset_idx)
        prev_triplets = triplets
        current_index += batch_size
        dataset_idx += 1
    print("✅ All datasets generated!")


if __name__ == "__main__":
    main()
