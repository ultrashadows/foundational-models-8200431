from pymongo import MongoClient, UpdateOne
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def embed_documents(model, collection, batch_size=256):
    update_ops = []
    embedded_count = 0
    skipped_count = 0
    batch_texts = []
    batch_ids = []

    cursor = collection.find({"embedding": {"$exists": False}}, {"_id": 1, "Texto Integral": 1, "embedding": 1})

    print("🚀 Starting streaming embedding loop...")
    for doc in tqdm(cursor, desc="Embedding documents"):
        if "embedding" in doc:
            skipped_count += 1
            continue

        if "Texto Integral" not in doc or not doc["Texto Integral"].strip():
            skipped_count += 1
            continue

        batch_texts.append(doc["Texto Integral"])
        batch_ids.append(doc["_id"])

        if len(batch_texts) == batch_size:
            embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()
            update_ops.extend(
                UpdateOne({"_id": doc_id}, {"$set": {"embedding": emb}})
                for doc_id, emb in zip(batch_ids, embeddings)
            )
            embedded_count += len(batch_texts)
            batch_texts, batch_ids = [], []

            if len(update_ops) >= 10000:
                print(f"📝 Writing {len(update_ops)} updates to MongoDB...")
                collection.bulk_write(update_ops)
                update_ops = []

    if batch_texts:
        embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()
        update_ops.extend(
            UpdateOne({"_id": doc_id}, {"$set": {"embedding": emb}})
            for doc_id, emb in zip(batch_ids, embeddings)
        )
        embedded_count += len(batch_texts)

    if update_ops:
        print(f"📝 Final write of {len(update_ops)} updates to MongoDB...")
        collection.bulk_write(update_ops)

    print(f"✅ Done: {embedded_count} embedded | {skipped_count} skipped")


if __name__ == '__main__':
    print("🔗 Connecting to MongoDB...")
    client = MongoClient("mongodb://localhost:27017/")
    db = client["database"]
    collection = db["pool"]

    print("🧠 Loading embedding model...")
    model = SentenceTransformer("paraphrase-mpnet-base-v2", device="cuda")

    embed_documents(model, collection)

    print("🏁 All documents embedded.")
