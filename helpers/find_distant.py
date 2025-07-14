import numpy as np
from pymongo import MongoClient
import faiss
from tqdm import tqdm


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

    # Normalize embeddings for cosine similarity
    normalized_embeddings = normalize_vectors(all_embeddings)

    # Build the FAISS index (inner product == cosine similarity if normalized)
    dim = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product
    index.add(normalized_embeddings)

    # Process only documents that don't have 'wrong'
    to_process = list(collection.find({"wrong": {"$exists": False}, "embedding": {"$exists": True}}, {"_id": 1, "embedding": 1}))
    total = len(to_process)

    print(f"Processing {total} documents...")

    for doc in tqdm(to_process):
        doc_id = doc["_id"]
        emb = np.array(doc["embedding"], dtype=np.float32).reshape(1, -1)
        emb = normalize_vectors(emb)

        # Query for most distant (lowest cosine similarity)
        similarities, indices = index.search(emb, k=len(all_ids))
        for i in indices[0]:
            if all_ids[i] != doc_id:
                most_distant_id = all_ids[i]
                break

        collection.update_one({"_id": doc_id}, {"$set": {"wrong": most_distant_id}})
