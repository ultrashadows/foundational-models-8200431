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
TRIPLETS_PER_DOC = 5
OVERLAP_PERCENT = 0.2

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


def find_best_and_worst_chunks_vectorized(query_embedding, correct_chunks, wrong_chunks, correct_embeddings, wrong_embeddings):
    if len(correct_chunks) == 0 or len(wrong_chunks) == 0:
        return None, None

    correct_scores = util.cos_sim(query_embedding, correct_embeddings)[0]
    wrong_scores = util.cos_sim(query_embedding, wrong_embeddings)[0]

    best_idx = correct_scores.argmax().item()
    worst_idx = wrong_scores.argmin().item()

    return correct_chunks[best_idx], wrong_chunks[worst_idx]


def get_last_dataset_index() -> int:
    files = os.listdir(DATASETS_DIR)
    indices = [int(f.split('_')[1].split('.')[0]) for f in files if f.startswith("dataset_") and f.endswith(".json")]
    return max(indices) if indices else 0


def process_batch(batch_objects: List[Dict], prev_triplets: List[Dict], dataset_idx: int):
    all_correct_chunks = set()
    all_wrong_chunks = set()
    all_queries = []
    object_data = []

    for obj in tqdm(batch_objects, desc=f"[#{dataset_idx}] Collecting data"):
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
            all_queries.append(query)

            object_data.append({
                "oid": oid,
                "pergunta": pergunta,
                "query": query,
                "correct_chunks": correct_chunks,
                "wrong_chunks": wrong_chunks
            })

    if not object_data:
        print(f"[#{dataset_idx}] ❌ No valid objects to process.")
        return []

    print(f"[#{dataset_idx}] 🔢 Precomputing embeddings...")
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
    for i, data in enumerate(tqdm(object_data, desc=f"[#{dataset_idx}] Generating triplets")):
        query_embedding = query_embeddings[i:i + 1]

        correct_indices = [correct_chunk_to_idx[chunk] for chunk in data["correct_chunks"] if chunk in correct_chunk_to_idx]
        wrong_indices = [wrong_chunk_to_idx[chunk] for chunk in data["wrong_chunks"] if chunk in wrong_chunk_to_idx]

        if not correct_indices or not wrong_indices:
            continue

        obj_correct_embeddings = correct_embeddings[correct_indices]
        obj_wrong_embeddings = wrong_embeddings[wrong_indices]

        positive, negative = find_best_and_worst_chunks_vectorized(
            query_embedding,
            data["correct_chunks"],
            data["wrong_chunks"],
            obj_correct_embeddings,
            obj_wrong_embeddings
        )

        if positive and negative:
            triplets.append({
                "doc_id": data["oid"],
                "question": data["pergunta"],
                "positive": positive,
                "negative": negative
            })

    if prev_triplets:
        overlap_count = int(OVERLAP_PERCENT * len(prev_triplets))
        triplets += random.sample(prev_triplets, min(overlap_count, len(prev_triplets)))

    output_file = os.path.join(DATASETS_DIR, f"dataset_{dataset_idx}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)

    print(f"[#{dataset_idx}] ✅ Saved {len(triplets)} triplets to {output_file}")
    return triplets


def main():
    print("📦 Loading dataset...")
    all_data = load_dataset()
    total_docs = len(all_data)

    dataset_idx = get_last_dataset_index() + 1
    current_index = 0
    prev_triplets = []

    while current_index < total_docs:
        if dataset_idx == 1:
            batch_size = FIRST_BATCH_SIZE
        else:
            batch_size = NEXT_BATCH_SIZE

        batch_objects = all_data[current_index:current_index + batch_size]
        if not batch_objects:
            break

        triplets = process_batch(batch_objects, prev_triplets, dataset_idx)

        prev_triplets = triplets
        current_index += batch_size
        dataset_idx += 1

    print("🏁 All datasets generated!")


if __name__ == "__main__":
    main()
