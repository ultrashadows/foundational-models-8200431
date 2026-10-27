"""Find, for each document, the most distant document in ``database.encoder`` (the ``wrong`` field).

This is the earlier negative-selection strategy. The final training data comes from
``helpers/triplet_generator.py`` and does not use ``wrong``.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from functools import cache

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from db import connect

MODEL_NAME = "paraphrase-mpnet-base-v2"
ENCODE_BATCH = 100

# Missing embeddings are computed once and written back into the shared document list,
# so concurrent callers wait for the first one instead of re-encoding the corpus.
_fill_lock = threading.Lock()


@cache
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def fill_embeddings(model, documents: list[dict], collection) -> tuple[list, list]:
    """Return (embeddings, ids) for every document that has text, encoding the ones that lack one.

    The two lists are index-aligned.
    """
    embeddings, ids = [], []
    batch = []

    def flush():
        batch_embeddings = model.encode([doc["Texto Integral"] for doc in batch]).tolist()
        for doc, embedding in zip(batch, batch_embeddings):
            doc["embedding"] = embedding
            collection.update_one({"_id": doc["_id"]}, {"$set": {"embedding": embedding}})
            embeddings.append(embedding)
            ids.append(doc["_id"])
        batch.clear()

    for doc in documents:
        if "embedding" in doc:
            embeddings.append(doc["embedding"])
            ids.append(doc["_id"])
        elif "Texto Integral" in doc:
            batch.append(doc)
            if len(batch) == ENCODE_BATCH:
                print(f"Encoded {len(ids) + len(batch)} documents...")
                flush()
    if batch:
        flush()

    return embeddings, ids


def embed(document: dict, all_documents: list[dict], collection=None):
    """Return the document in ``all_documents`` with the lowest cosine similarity to ``document``."""
    collection = collection if collection is not None else connect()["database"]["encoder"]
    model = get_model()

    if "embedding" in document:
        target_embedding = np.array(document["embedding"])
    else:
        target_embedding = model.encode([document["Texto Integral"]])[0]
        collection.update_one({"_id": document["_id"]}, {"$set": {"embedding": target_embedding.tolist()}})

    with _fill_lock:
        embeddings, ids = fill_embeddings(model, all_documents, collection)

    if not embeddings:
        print("No embeddings found in the collection.")
        return None

    similarities = cosine_similarity([target_embedding], np.array(embeddings))
    return collection.find_one({"_id": ids[int(np.argmin(similarities))]})


def process_document(index, total, document, documents, collection, lock):
    try:
        print(f"{index + 1}/{total}")
        most_distant = embed(document, documents, collection)
        with lock:
            collection.update_one({"_id": document["_id"]}, {"$set": {"wrong": most_distant["_id"]}})
    except Exception as e:
        print(f"Error processing document {index}: {e}")


def main():
    collection = connect()["database"]["encoder"]
    all_documents = list(collection.find({}))
    pending = [doc for doc in all_documents if "wrong" not in doc]
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=20) as executor:
        for index, document in enumerate(pending):
            executor.submit(process_document, index, len(pending), document, all_documents, collection, lock)


if __name__ == "__main__":
    main()
