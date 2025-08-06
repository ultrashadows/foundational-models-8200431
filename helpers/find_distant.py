import numpy as np
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


if __name__ == '__main__':
    client = MongoClient("mongodb://localhost:27017/")
    db = client["database"]
    encoder_col = db["test"]
    pool_col = db["test"]

    encoder_docs = list(encoder_col.find({"embedding": {"$exists": True}}, {"_id": 1, "embedding": 1}))

    print("Loading pool documents...")
    pool_docs = list(pool_col.find({"embedding": {"$exists": True}}, {"_id": 1, "embedding": 1}))

    if not encoder_docs or not pool_docs:
        print("❌ Missing embeddings in 'encoder' or 'pool'.")
        exit(1)

    print(f"🔍 {len(encoder_docs)} encoder docs | 🎯 {len(pool_docs)} pool docs")

    pool_ids = [doc["_id"] for doc in pool_docs]
    pool_embeddings = np.array([doc["embedding"] for doc in pool_docs], dtype=np.float32)
    pool_embeddings = normalize(pool_embeddings)

    used_wrong_ids = set()
    id_to_index = {doc_id: idx for idx, doc_id in enumerate(pool_ids)}

    batch_size = 256
    batch_updates = []

    print("Processing encoder documents...")
    for i in tqdm(range(0, len(encoder_docs), batch_size), desc="Processing batches"):
        batch_docs = encoder_docs[i:i + batch_size]

        batch_embeddings = np.array([doc["embedding"] for doc in batch_docs], dtype=np.float32)
        batch_embeddings = normalize(batch_embeddings)

        similarities = batch_embeddings @ pool_embeddings.T

        for j, doc in enumerate(batch_docs):
            sim_row = similarities[j]
            global_index = i + j

            candidate_indices = np.where((sim_row >= -0.3) & (sim_row <= 0.2))[0]
            candidate_indices = [idx for idx in candidate_indices if pool_ids[idx] not in used_wrong_ids]

            if candidate_indices:
                selected_idx = np.random.choice(candidate_indices)
                wrong_id = pool_ids[selected_idx]
                used_wrong_ids.add(wrong_id)
            else:
                sorted_indices = np.argsort(sim_row)
                fallback_idx = None

                for idx in sorted_indices:
                    if pool_ids[idx] not in used_wrong_ids:
                        fallback_idx = idx
                        break

                if fallback_idx is not None:
                    fallback_sim = sim_row[fallback_idx]
                    wrong_id = pool_ids[fallback_idx]
                    used_wrong_ids.add(wrong_id)
                    print(f"⚠️ Fallback at index {global_index}: no available candidates in [-0.3, 0.2]. "
                          f"Using most dissimilar (sim={fallback_sim:.3f}) → {wrong_id}")
                else:
                    print(f"❌ No available unused 'wrong' documents for encoder index {global_index}. Skipping.")
                    continue

            batch_updates.append(
                UpdateOne({"_id": doc["_id"]}, {"$set": {"wrong": wrong_id}})
            )

        if len(batch_updates) >= 5000 or i + batch_size >= len(encoder_docs):
            if batch_updates:
                encoder_col.bulk_write(batch_updates)
                batch_updates = []

    print("✅ All encoder docs updated with unique cosine-distant 'wrong' examples.")
