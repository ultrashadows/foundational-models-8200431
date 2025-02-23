import numpy as np
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def fetch_documents(collection, document_id):
    return list(collection.find({'_id': {'$ne': document_id}}))


def get_embedding(model: SentenceTransformer, documents: any, collection, embeddings: list[any]):
    batch = []
    ids = []

    for i, doc in enumerate(documents):
        if "embedding" in doc:
            embeddings.append(doc["embedding"])
            continue

        if "Decisão Texto Integral" not in doc:
            continue

        batch.append(doc["Decisão Texto Integral"])
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


def embed(document):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["files"]
    collection = db["jtrl"]

    document_id = document["_id"]
    print("Fetching all documents...")
    all_documents = fetch_documents(collection, document_id)

    print("Embedding documents...")
    model = SentenceTransformer("paraphrase-mpnet-base-v2")

    if "embedding" not in document:
        target_embedding = model.encode([document["Decisão Texto Integral"]])[0]
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
