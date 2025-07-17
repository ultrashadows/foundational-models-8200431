import numpy as np
from pymongo import MongoClient
import faiss
from tqdm import tqdm

TARGET_MARGIN = 0.2
TOP_K = 10000


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


if __name__ == '__main__':
    client = MongoClient("mongodb://localhost:27017/")
    db = client["database"]
    collection = db["encoder"]

    all_docs = list(collection.find({"embedding": {"$exists": True}}, {"_id": 1, "embedding": 1}))
    all_ids = [doc["_id"] for doc in all_docs]
    all_embeddings = np.array([doc["embedding"] for doc in all_docs], dtype=np.float32)

    normalized_embeddings = normalize_vectors(all_embeddings)

    dim = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normalized_embeddings)

    id_to_index = {doc_id: idx for idx, doc_id in enumerate(all_ids)}

    to_process = list(collection.find(
        {"wrong": {"$exists": False}, "embedding": {"$exists": True}},
        {"_id": 1, "embedding": 1}
    ))
    total = len(to_process)
    print(f"Processing {total} documents...")

    fallback_count = 0
    total_diff = 0.0
    processed_count = 0

    with tqdm(to_process) as pbar:
        for doc in pbar:
            doc_id = doc["_id"]
            emb = np.array(doc["embedding"], dtype=np.float32).reshape(1, -1)
            emb = normalize_vectors(emb)

            similarities, indices = index.search(emb, k=TOP_K + 1)
            similarities = similarities[0]
            indices = indices[0]

            positive_similarity = None
            for sim, idx in zip(similarities, indices):
                if all_ids[idx] != doc_id:
                    positive_similarity = sim
                    break

            if positive_similarity is None:
                print(f"⚠️ Could not find a positive for {doc_id}. Skipping.")
                continue

            target_sim = positive_similarity - TARGET_MARGIN

            best_sim = None
            best_diff = float('inf')
            selected_wrong = None  # Initialize here to avoid UnboundLocalError

            # Find the wrong candidate closest to target margin
            for sim, idx in zip(similarities, indices):
                candidate_id = all_ids[idx]
                if candidate_id == doc_id:
                    continue
                if target_sim <= sim < positive_similarity:
                    diff = positive_similarity - sim
                    abs_diff = abs(diff - TARGET_MARGIN)
                    if abs_diff < best_diff:
                        best_diff = abs_diff
                        best_sim = sim
                        selected_wrong = candidate_id

            # Fallback if no suitable candidate found
            if not selected_wrong:
                fallback_count += 1
                for sim, idx in zip(reversed(similarities), reversed(indices)):
                    candidate_id = all_ids[idx]
                    if candidate_id != doc_id:
                        selected_wrong = candidate_id
                        best_sim = sim  # assign best_sim in fallback as well
                        print(f"[Fallback] {doc_id} | Positive Sim: {positive_similarity:.4f} | Fallback Neg Sim: {sim:.4f}")
                        break

            collection.update_one({"_id": doc_id}, {"$set": {"wrong": selected_wrong}})

            if selected_wrong and best_sim is not None:
                diff = positive_similarity - best_sim
                total_diff += diff
                processed_count += 1

            avg_diff = total_diff / processed_count if processed_count > 0 else 0
            pbar.set_postfix(avg_diff=f"{avg_diff:.4f}", fallbacks=fallback_count)

    print(f"\n✅ Done. Fallbacks used for {fallback_count}/{total} documents.")
    print(f"Average difference (positive_sim - wrong_sim): {avg_diff:.4f}")
