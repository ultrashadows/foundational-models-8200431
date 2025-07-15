import json
import os
import pickle
import random
from tqdm import tqdm
from typing import List, Dict
import torch

from sentence_transformers import SentenceTransformer, util

DATASET_PATH = "dataset.json"
CACHE_DIR = "cache"
OUTPUT_PATH = "triplet_output.json"
SAMPLE_SIZE = 2000
BATCH_SIZE = 100

model = SentenceTransformer('intfloat/multilingual-e5-base', device="cuda")


def stream_json_objects(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = json.load(f)
        for obj in raw:
            yield obj


def load_random_objects(filepath: str, k: int) -> List[Dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
        total = len(all_data)
        sampled = random.sample(all_data, min(k, total))
    return sampled


def load_cache(oid: str) -> Dict:
    with open(os.path.join(CACHE_DIR, f"{oid}.pkl"), "rb") as f:
        return pickle.load(f)


def precompute_chunk_embeddings(all_chunks: List[str], batch_size: int = BATCH_SIZE):
    embeddings = []
    chunk_texts = [f"passage: {chunk}" for chunk in all_chunks]

    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        embeddings.append(batch_embeddings)

    return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, model.get_sentence_embedding_dimension())


def find_best_and_worst_chunks_vectorized(query_embedding, correct_chunks: List[str], wrong_chunks: List[str],
                                          correct_embeddings, wrong_embeddings):
    if len(correct_chunks) == 0 or len(wrong_chunks) == 0:
        return None, None

    correct_scores = util.cos_sim(query_embedding, correct_embeddings)[0]
    wrong_scores = util.cos_sim(query_embedding, wrong_embeddings)[0]

    best_idx = correct_scores.argmax().item()
    worst_idx = wrong_scores.argmin().item()

    return correct_chunks[best_idx], wrong_chunks[worst_idx]


def main():
    selected_objects = load_random_objects(DATASET_PATH, SAMPLE_SIZE)

    print("📊 Collecting all chunks and queries...")
    all_correct_chunks = set()
    all_wrong_chunks = set()
    all_queries = []
    object_data = []

    for obj in tqdm(selected_objects, desc="Collecting data"):
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

    print("🔢 Precomputing embeddings...")
    all_correct_chunks = list(all_correct_chunks)
    all_wrong_chunks = list(all_wrong_chunks)

    correct_chunk_to_idx = {chunk: i for i, chunk in enumerate(all_correct_chunks)}
    wrong_chunk_to_idx = {chunk: i for i, chunk in enumerate(all_wrong_chunks)}

    print("Computing correct chunk embeddings...")
    correct_embeddings = precompute_chunk_embeddings(all_correct_chunks)

    print("Computing wrong chunk embeddings...")
    wrong_embeddings = precompute_chunk_embeddings(all_wrong_chunks)

    print("Computing query embeddings...")
    query_texts = [f"query: {query}" for query in all_queries]
    query_embeddings = []
    for i in range(0, len(query_texts), BATCH_SIZE):
        batch = query_texts[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        query_embeddings.append(batch_embeddings)

    query_embeddings = torch.cat(query_embeddings, dim=0) if query_embeddings else torch.empty(0,
                                                                                               model.get_sentence_embedding_dimension())

    print("🎯 Generating triplets...")
    triplets = []

    for i, data in enumerate(tqdm(object_data, desc="Processing triplets")):
        query_embedding = query_embeddings[i:i + 1]

        correct_indices = [correct_chunk_to_idx[chunk] for chunk in data["correct_chunks"] if
                           chunk in correct_chunk_to_idx]
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

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated {len(triplets)} triplets and saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
