import json
import os
import pickle
import random
from tqdm import tqdm
from typing import List, Dict
import torch
from sentence_transformers import SentenceTransformer, util

DATASET_PATH = "../encoder/dataset.json"
CACHE_DIR = "../encoder/cache"
DATASETS_DIR = "../encoder/datasets"
BATCH_SIZE = 100

FIRST_BATCH_SIZE = 2000
NEXT_BATCH_SIZE = 1600
OVERLAP_PERCENT = 0.2
HARD_NEGATIVE_K = 5
BASE_MARGIN = 0.2

os.makedirs(DATASETS_DIR, exist_ok=True)
model = SentenceTransformer('intfloat/multilingual-e5-base', device="cuda")


def load_dataset() -> List[Dict]:
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_cache(oid: str) -> Dict:
    with open(os.path.join(CACHE_DIR, f"{oid}.pkl"), "rb") as f:
        return pickle.load(f)


def precompute_chunk_embeddings(all_chunks: List[str]) -> torch.Tensor:
    embeddings = []
    chunk_texts = [f"passage: {chunk}" for chunk in all_chunks]

    for i in range(0, len(chunk_texts), BATCH_SIZE):
        batch = chunk_texts[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        embeddings.append(batch_embeddings)

    return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, model.get_sentence_embedding_dimension())


def find_best_and_worst_chunks_vectorized(
    query_embedding,
    correct_chunks,
    wrong_chunks,
    correct_embeddings,
    wrong_embeddings,
    hard_negative_k=5,
    base_margin=0.2,
    dataset_idx=1,
    item_index=None
):
    if len(correct_chunks) == 0 or len(wrong_chunks) == 0:
        return None, None

    correct_scores = util.cos_sim(query_embedding, correct_embeddings)[0]
    wrong_scores = util.cos_sim(query_embedding, wrong_embeddings)[0]

    pos_idx = torch.topk(correct_scores, 1).indices[0].item()
    positive_sim = correct_scores[pos_idx].item()
    positive = correct_chunks[pos_idx]

    adaptive_margin = max(base_margin, min(base_margin + 0.01 * dataset_idx, 0.25))
    lower = positive_sim - adaptive_margin - 0.05
    upper = positive_sim - adaptive_margin + 0.05

    mask = (wrong_scores < positive_sim) & (wrong_scores >= lower) & (wrong_scores <= upper)
    candidate_neg_indices = torch.where(mask)[0]

    best_idx = None

    if len(candidate_neg_indices) > 0:
        candidate_scores = wrong_scores[candidate_neg_indices]
        actual_margins = positive_sim - candidate_scores
        abs_diff = torch.abs(actual_margins - adaptive_margin)
        best_idx = candidate_neg_indices[abs_diff.argmin()]
    else:
        semi_hard_margin = adaptive_margin - 0.05
        alt_mask = (wrong_scores < positive_sim) & (wrong_scores >= (positive_sim - semi_hard_margin))
        alt_neg_indices = torch.where(alt_mask)[0]
        if len(alt_neg_indices) > 0:
            best_idx = alt_neg_indices[torch.argmin(wrong_scores[alt_neg_indices])]
        else:
            topk = min(hard_negative_k + dataset_idx, len(wrong_scores))
            topk_indices = torch.topk(wrong_scores, k=topk, largest=True).indices
            for idx in topk_indices:
                if wrong_scores[idx].item() < positive_sim:
                    best_idx = idx
                    break

    if best_idx is None:
        sorted_scores, sorted_indices = torch.sort(wrong_scores)

        cutoff = max(3, len(sorted_indices) // 4)
        candidate_indices = sorted_indices[:cutoff]

        if len(candidate_indices) > 1:
            best_idx = random.choice(candidate_indices[1:])
        else:
            best_idx = sorted_indices[0]

        if item_index is not None:
            print(f"⚠️ [Soft Fallback] No margin-fit negative — used easier fallback for item {item_index}")

    negative = wrong_chunks[best_idx]
    return positive, negative


def get_last_dataset_index() -> int:
    files = os.listdir(DATASETS_DIR)
    indices = [int(f.split('_')[1].split('.')[0]) for f in files if f.startswith("dataset_") and f.endswith(".json")]
    return max(indices) if indices else 0


def process_batch(batch_objects: List[Dict], prev_triplets: List[Dict], dataset_idx: int):
    all_correct_chunks = set()
    all_wrong_chunks = set()
    object_data = []

    print(f"[# {dataset_idx}] Collecting data from batch ({len(batch_objects)} docs)...")
    for obj in tqdm(batch_objects, desc=f"[# {dataset_idx}] Collecting data"):
        oid = obj["_id"]["$oid"]
        try:
            cache = load_cache(oid)
        except FileNotFoundError:
            continue

        correct_chunks = cache.get("correct_chunks", [])
        wrong_chunks = cache.get("wrong_chunks", [])

        if not correct_chunks or not wrong_chunks:
            continue

        all_correct_chunks.update(correct_chunks)
        all_wrong_chunks.update(wrong_chunks)

        for result in obj.get("results", []):
            pergunta = result["pergunta"]
            resposta = result["resposta"]
            query = f"{pergunta} {resposta}"
            object_data.append({
                "oid": oid,
                "pergunta": pergunta,
                "query": query,
                "correct_chunks": correct_chunks,
                "wrong_chunks": wrong_chunks
            })

    if not object_data:
        print(f"[# {dataset_idx}] ❌ No valid objects to process.")
        return []

    print(f"[# {dataset_idx}] Precomputing embeddings for chunks...")
    all_correct_chunks = list(all_correct_chunks)
    all_wrong_chunks = list(all_wrong_chunks)

    correct_chunk_to_idx = {chunk: i for i, chunk in enumerate(all_correct_chunks)}
    wrong_chunk_to_idx = {chunk: i for i, chunk in enumerate(all_wrong_chunks)}

    correct_embeddings = precompute_chunk_embeddings(all_correct_chunks)
    wrong_embeddings = precompute_chunk_embeddings(all_wrong_chunks)

    query_texts = [f"query: {q['query']}" for q in object_data]
    query_embeddings = []
    for i in range(0, len(query_texts), BATCH_SIZE):
        batch = query_texts[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        query_embeddings.append(batch_embeddings)
    query_embeddings = torch.cat(query_embeddings, dim=0)

    triplets = []
    skipped_triplets = 0

    print(f"[# {dataset_idx}] Generating triplets...")
    for i, data in enumerate(tqdm(object_data, desc=f"[# {dataset_idx}] Triplet generation")):
        query_embedding = query_embeddings[i:i + 1]

        correct_indices = [correct_chunk_to_idx[chunk] for chunk in data["correct_chunks"] if chunk in correct_chunk_to_idx]
        wrong_indices = [wrong_chunk_to_idx[chunk] for chunk in data["wrong_chunks"] if chunk in wrong_chunk_to_idx]

        if not correct_indices or not wrong_indices:
            skipped_triplets += 1
            continue

        obj_correct_embeddings = correct_embeddings[correct_indices]
        obj_wrong_embeddings = wrong_embeddings[wrong_indices]

        positive, negative = find_best_and_worst_chunks_vectorized(
            query_embedding,
            data["correct_chunks"],
            data["wrong_chunks"],
            obj_correct_embeddings,
            obj_wrong_embeddings,
            hard_negative_k=HARD_NEGATIVE_K,
            base_margin=BASE_MARGIN,
            dataset_idx=dataset_idx,
            item_index=i
        )

        if positive is None or negative is None:
            skipped_triplets += 1
            continue

        triplets.append({
            "doc_id": data["oid"],
            "question": data["pergunta"],
            "positive": positive,
            "negative": negative
        })

    if prev_triplets:
        print(f"[# {dataset_idx}] Selecting best overlap triplets...")
        overlap_count = int(OVERLAP_PERCENT * len(prev_triplets))
        if overlap_count > 0:
            triplet_texts = [(f"query: {t['question']}", f"passage: {t['positive']}", f"passage: {t['negative']}") for t
                             in prev_triplets]

            anchor_embs = model.encode([t[0] for t in triplet_texts], convert_to_tensor=True, show_progress_bar=False)
            pos_embs = model.encode([t[1] for t in triplet_texts], convert_to_tensor=True, show_progress_bar=False)
            neg_embs = model.encode([t[2] for t in triplet_texts], convert_to_tensor=True, show_progress_bar=False)

            pos_sims = util.cos_sim(anchor_embs, pos_embs).diagonal()
            neg_sims = util.cos_sim(anchor_embs, neg_embs).diagonal()
            margins = neg_sims - pos_sims

            ideal_margin = 0.2
            margin_dist = torch.abs(margins - ideal_margin)
            best_indices = torch.topk(-margin_dist, overlap_count).indices

            triplets += [prev_triplets[i] for i in best_indices]

    print(f"[# {dataset_idx}] Created {len(triplets)} triplets, skipped {skipped_triplets} due to filtering.")
    output_file = os.path.join(DATASETS_DIR, f"dataset_{dataset_idx}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)

    print(f"[# {dataset_idx}] Saved triplets to {output_file}")
    return triplets


def main():
    print("Loading full dataset...")
    all_data = load_dataset()
    total_docs = len(all_data)

    dataset_idx = get_last_dataset_index() + 1
    current_index = (FIRST_BATCH_SIZE if dataset_idx > 1 else 0) + (dataset_idx - 2) * NEXT_BATCH_SIZE

    prev_triplets = []
    if dataset_idx > 1:
        prev_file = os.path.join(DATASETS_DIR, f"dataset_{dataset_idx - 1}.json")
        if os.path.exists(prev_file):
            with open(prev_file, "r", encoding="utf-8") as f:
                prev_triplets = json.load(f)

    while current_index < total_docs:
        batch_size = FIRST_BATCH_SIZE if dataset_idx == 1 else NEXT_BATCH_SIZE

        batch_objects = all_data[current_index:current_index + batch_size]
        if not batch_objects:
            break

        triplets = process_batch(batch_objects, prev_triplets, dataset_idx)

        prev_triplets = triplets
        current_index += batch_size
        dataset_idx += 1

    print("All datasets generated!")


if __name__ == "__main__":
    main()
