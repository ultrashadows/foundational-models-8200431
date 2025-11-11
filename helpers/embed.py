import threading

import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor


def get_embedding(model: SentenceTransformer, documents: any, collection, embeddings: list[any]):
    batch = []
    ids = []

    for i, doc in enumerate(documents):
        if "embedding" in doc:
            embeddings.append(doc["embedding"])
            continue

        if "Texto Integral" not in doc:
            continue

        batch.append(doc["Texto Integral"])
        ids.append(doc["_id"])

        if (i + 1) % 100 == 0:
            print(f"Processing document {i + 1}...")
            batch_embeddings = model.encode(batch).tolist()
            embeddings.extend(batch_embeddings)

            for doc_id, embedding in zip(ids, batch_embeddings):
                collection.update_one({"_id": doc_id}, {"$set": {"embedding": embedding}})

            batch.clear()
            ids.clear()

    if batch:
        batch_embeddings = model.encode(batch).tolist()
        embeddings.extend(batch_embeddings)
        for doc_id, embedding in zip(ids, batch_embeddings):
            collection.update_one({"_id": doc_id}, {"$set": {"embedding": embedding}})

    return embeddings


def embed(document, all_documents):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["database"]
    collection = db["encoder"]

    document_id = document["_id"]

    print("Embedding documents...")
    model = SentenceTransformer("paraphrase-mpnet-base-v2")

    if "embedding" not in document:
        target_embedding = model.encode([document["Texto Integral"]])[0]
        collection.update_one({"_id": document_id}, {"$set": {"embedding": target_embedding.tolist()}})
    else:
        target_embedding = np.array(document["embedding"])

    all_embeddings = np.array(get_embedding(model, all_documents, collection, []))

    if all_embeddings.size == 0:
        print("No embeddings found in the collection.")
        return None

    print("Finding most distant document...")
    similarities = cosine_similarity([target_embedding], all_embeddings)
    most_distant_index = np.argmin(similarities)
    most_distant_document = collection.find_one({"_id": all_documents[most_distant_index]["_id"]})

    return most_distant_document


def process_document(index, document, documents, collection, lock):
    try:
        print(f"{index + 1}/{len(documents)}")
        most_distant = embed(document, documents)
        with lock:
            collection.update_one(
                {"_id": document["_id"]},
                {"$set": {"wrong": most_distant["_id"]}}
            )
    except Exception as e:
        print(f"Error processing document {index}: {e}")


if __name__ == '__main__':
    client = MongoClient("mongodb://localhost:27017/")
    db = client["database"]
    collection = db["encoder"]

    all_documents = list(collection.find({}))
    documents = list(collection.find({"wrong": {"$exists": False}}))
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=20) as executor:
        for index, document in enumerate(documents):
            executor.submit(process_document, index, document, all_documents, collection, lock)