import numpy as np
from pymongo import MongoClient, UpdateOne, WriteConcern
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def ensure_embeddings(col, text_field="Texto Integral", batch_fetch=20000, encode_bs=256):
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    query = {"embedding": {"$exists": False}, text_field: {"$exists": True}}
    total = col.count_documents(query)
    if total == 0:
        print("✅ All documents already have embeddings.")
        return
    cursor = col.find(query, {"_id": 1, text_field: 1}).batch_size(batch_fetch)
    ids, texts, ops = [], [], []
    pbar = tqdm(total=total, desc="Generating embeddings", unit="doc")
    for doc in cursor:
        ids.append(doc["_id"])
        texts.append(doc[text_field])
        if len(texts) >= encode_bs:
            embs = model.encode(texts, batch_size=encode_bs, convert_to_numpy=True, show_progress_bar=False)
            for _id, emb in zip(ids, embs):
                ops.append(UpdateOne({"_id": _id}, {"$set": {"embedding": emb.astype(np.float32).tolist()}}))
            if ops:
                col.with_options(write_concern=WriteConcern(w=0, j=False)).bulk_write(ops, ordered=False)
            pbar.update(len(ids))
            ids, texts, ops = [], [], []
    if texts:
        embs = model.encode(texts, batch_size=encode_bs, convert_to_numpy=True, show_progress_bar=False)
        for _id, emb in zip(ids, embs):
            ops.append(UpdateOne({"_id": _id}, {"$set": {"embedding": emb.astype(np.float32).tolist()}}))
        if ops:
            col.with_options(write_concern=WriteConcern(w=0, j=False)).bulk_write(ops, ordered=False)
        pbar.update(len(ids))
    pbar.close()


if __name__ == '__main__':
    client = MongoClient("mongodb://localhost:27017/")
    db = client["database"]
    col = db["encoder"]
    ensure_embeddings(col)
