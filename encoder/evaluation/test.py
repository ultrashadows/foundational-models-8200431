import numpy as np
import pandas as pd
import torch
import os
import json
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from math import ceil

chunk_size = 512
stride = 32
top_k = 5
MAX_LEN = 512

model = SentenceTransformer("PORTULAN/albertina-1b5-portuguese-ptpt-encoder", device='cuda')
model = model.half()
tokenizer = model.tokenizer


def sliding_window_chunk(text, tokenizer, max_len=512, stride=32):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(tokens), stride):
        end = start + max_len
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            break
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end >= len(tokens):
            break
    return chunks


def chunk_worker(item):
    doc_id, doc_text = item
    chunks = sliding_window_chunk(doc_text, tokenizer, chunk_size, stride)
    return [(doc_id, chunk) for chunk in chunks]


def load_data():
    with open("evaluation/test.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    documents, questions = {}, []
    print("Loading documents and questions from JSON...")
    for doc in tqdm(raw_data, desc="Loading from JSON"):
        doc_id = str(doc["_id"])
        text = doc.get("Texto Integral", "")
        if not text.strip():
            continue
        documents[doc_id] = text
        for i, result in enumerate(doc.get("results", [])):
            question = result.get("pergunta", "").strip()
            if question:
                questions.append({
                    "id": f"{doc_id}_{i}",
                    "text": question,
                    "relevant_doc": doc_id
                })
    return documents, questions


def dynamic_encode(batch_chunks, max_bs):
    bs = max_bs
    while bs > 0:
        print(f"Trying batch size: {bs}")
        torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                embeddings = model.encode(
                    batch_chunks,
                    batch_size=bs,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            return np.asarray(embeddings, dtype=np.float32)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"OOM - retrying with batch_size={bs // 2}")
                bs = bs // 2
            else:
                raise
    raise RuntimeError("Failed to encode batch, even with batch size = 1")


def build_chunk_database(documents):
    cache_dir = "evaluation/cache/chunks"
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "chunk_metadata.pkl")

    print("Chunking documents...")
    results = [chunk_worker(item) for item in tqdm(documents.items(), desc="Chunking")]

    chunk_texts = []
    chunk_metadata = []
    for doc_chunks in results:
        for doc_id, chunk_text in doc_chunks:
            chunk_metadata.append({
                "chunk_id": len(chunk_texts),
                "doc_id": doc_id,
                "text": chunk_text
            })
            chunk_texts.append(chunk_text)

    with open(meta_path, "wb") as f:
        pickle.dump(chunk_metadata, f)

    total_chunks = len(chunk_texts)
    base_batch_size = 512
    n_batches = ceil(total_chunks / base_batch_size)

    print(f"Total Chunks: {total_chunks} | Batch Size: {base_batch_size} | Batches: {n_batches}")
    print("Encoding and saving in batches...")

    for batch_idx in tqdm(range(n_batches), desc="Encoding batches"):
        batch_file = os.path.join(cache_dir, f"batch_{batch_idx}.npy")
        if os.path.exists(batch_file):
            continue
        start = batch_idx * base_batch_size
        end = min((batch_idx + 1) * base_batch_size, total_chunks)
        batch_chunks = chunk_texts[start:end]
        embeddings = dynamic_encode(batch_chunks, max_bs=16)
        np.save(batch_file, embeddings)
        torch.cuda.empty_cache()

    all_embeddings = []
    for batch_idx in range(n_batches):
        batch_file = os.path.join(cache_dir, f"batch_{batch_idx}.npy")
        emb = np.load(batch_file).astype(np.float16)
        all_embeddings.append(emb)

    chunk_embeddings = np.vstack(all_embeddings)
    return chunk_metadata, torch.tensor(chunk_embeddings, dtype=torch.float16, device='cuda')


def compute_similarities_vectorized(query_embeddings, chunk_embeddings):
    return util.pytorch_cos_sim(query_embeddings, chunk_embeddings)


def batch_retrieve(questions, chunk_metadata, chunk_embeddings, batch_size=512):
    cache_dir = "evaluation/cache/results"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "retrieval_cache.jsonl")

    if os.path.exists(cache_file):
        print("Loading retrieval results from cache...")
        with open(cache_file, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    print("Preparing chunk-to-doc mappings...")
    chunk_doc_ids = np.array([meta["doc_id"] for meta in chunk_metadata])
    unique_doc_ids = np.unique(chunk_doc_ids)
    doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(unique_doc_ids)}
    doc_indices = np.array([doc_id_to_idx[doc] for doc in chunk_doc_ids])

    for i in tqdm(range(0, len(questions), batch_size), desc="Retrieving (batched)"):
        batch_questions = questions[i:i + batch_size]
        batch_texts = [q["text"] for q in batch_questions]

        query_embeddings = model.encode(
            batch_texts,
            batch_size=16,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        similarities = compute_similarities_vectorized(query_embeddings, chunk_embeddings)
        batch_results = []

        for j, question in enumerate(batch_questions):
            sim_vec = similarities[j].cpu().numpy()
            doc_scores_np = np.zeros(len(unique_doc_ids), dtype=np.float32)
            np.maximum.at(doc_scores_np, doc_indices, sim_vec)
            sorted_indices = np.argsort(-doc_scores_np)
            ranked_docs = [unique_doc_ids[k] for k in sorted_indices[:top_k]]

            batch_results.append({
                "question_id": question["id"],
                "relevant_doc": question["relevant_doc"],
                "retrieved_docs": ranked_docs,
                "ranked_docs": [unique_doc_ids[k] for k in sorted_indices]
            })

        with open(cache_file, "a", encoding="utf-8") as f:
            for r in batch_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(cache_file, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def compute_metrics_vectorized(results):
    n_queries = len(results)
    hits, precisions, recalls, f1s, mrrs, aps, r_precisions, rbps, dcgs, ndcgs = (
        np.zeros(n_queries) for _ in range(10)
    )
    p = 0.8

    for i, res in enumerate(results):
        relevant = res["relevant_doc"]
        ranked = res["ranked_docs"]
        binary_relevance = np.array([1 if doc == relevant else 0 for doc in ranked[:top_k]])

        hits[i] = 1 if relevant in ranked[:top_k] else 0
        precisions[i] = np.sum(binary_relevance) / top_k
        recalls[i] = np.sum(binary_relevance)
        f1s[i] = 2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i]) if precisions[i] + recalls[i] > 0 else 0
        try:
            rank = ranked.index(relevant)
            mrrs[i] = 1 / (rank + 1)
        except ValueError:
            pass

        r_precisions[i] = np.sum(binary_relevance)
        aps[i] = sum((np.sum(binary_relevance[:j + 1]) / (j + 1)) for j in range(top_k) if binary_relevance[j])
        rbps[i] = (1 - p) * np.sum((p ** np.arange(top_k)) * binary_relevance)
        dcgs[i] = np.sum(binary_relevance / np.log2(np.arange(2, top_k + 2)))
        ndcgs[i] = dcgs[i] / (1 / np.log2(2)) if relevant in ranked[:top_k] else 0

    return {
        'hits': hits, 'precisions': precisions, 'recalls': recalls, 'f1s': f1s,
        'mrrs': mrrs, 'aps': aps, 'r_precisions': r_precisions,
        'rbps': rbps, 'dcgs': dcgs, 'ndcgs': ndcgs
    }


def main():
    documents, questions = load_data()
    print(f"Loaded {len(documents)} documents and {len(questions)} questions")

    chunk_metadata, chunk_embeddings = build_chunk_database(documents)
    results = batch_retrieve(questions, chunk_metadata, chunk_embeddings)
    metrics = compute_metrics_vectorized(results)

    metric_definitions = [
        ("Hits@5", "Is the correct doc in top 5?", "Higher", "0-1", np.mean(metrics['hits'])),
        ("HitRate@5", "Same as Hits@5, redundancy", "Higher", "0-1", np.mean(metrics['hits'])),
        ("Precision@5", "Precision in top 5 results", "Higher", "0-1", np.mean(metrics['precisions'])),
        ("Recall@5", "Recall out of relevant items", "Higher", "0-1", np.mean(metrics['recalls'])),
        ("F1@5", "Harmonic mean of precision and recall", "Higher", "0-1", np.mean(metrics['f1s'])),
        ("R-Precision", "Precision at R (relevant = 1)", "Higher", "0-1", np.mean(metrics['r_precisions'])),
        ("RBP@5", "Rank Biased Precision with p=0.8", "Higher", "0-1", np.mean(metrics['rbps'])),
        ("MRR@5", "Mean Reciprocal Rank at 5", "Higher", "0-1", np.mean(metrics['mrrs'])),
        ("MAP@5", "Mean Average Precision at 5", "Higher", "0-1", np.mean(metrics['aps'])),
        ("DCG@5", "Discounted Cumulative Gain at 5", "Higher", "0-~", np.mean(metrics['dcgs'])),
        ("NDCG@5", "Normalized DCG at 5", "Higher", "0-1", np.mean(metrics['ndcgs'])),
    ]

    df_metrics = pd.DataFrame(metric_definitions, columns=["Test", "Explanation", "Better", "Range", "Result"])
    df_metrics.to_csv("evaluation/ir_metrics_results.csv", index=False)
    print("\nEvaluation Results:\n" + "=" * 80)
    print(df_metrics.to_string(index=False, float_format='%.4f'))

    detailed_results = pd.DataFrame([
        {
            'question_id': r['question_id'],
            'relevant_doc': r['relevant_doc'],
            'retrieved_docs': ','.join(r['retrieved_docs']),
            'hit': 1 if r['relevant_doc'] in r['retrieved_docs'] else 0
        }
        for r in results
    ])
    detailed_results.to_csv("evaluation/detailed_results.csv", index=False)
    print(f"\nSaved results to evaluation/")


if __name__ == "__main__":
    main()
