"""Report margin statistics (pos_sim - neg_sim) of a triplet dataset.

Uses the values stored by ``helpers/triplet_generator.py`` and recomputes them on the first 500
triplets with the generator's embedding model.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, util

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "datasets" / "dataset_1.json"


def analyze_dataset_margins(dataset_path, model_name="intfloat/multilingual-e5-base"):
    print(f"Analyzing dataset: {dataset_path}")
    print(f"Using model: {model_name}")

    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} triplets")

    # Basic statistics from stored values
    stored_margins = [item.get('margin', 0) for item in data if 'margin' in item]
    stored_pos_sims = [item.get('pos_sim', 0) for item in data if 'pos_sim' in item]
    stored_neg_sims = [item.get('neg_sim', 0) for item in data if 'neg_sim' in item]

    print("\n=== STORED VALUES ANALYSIS ===")
    if stored_margins:
        print(f"Stored margins - Mean: {np.mean(stored_margins):.4f}, Std: {np.std(stored_margins):.4f}")
        print(f"Stored margins - Min: {np.min(stored_margins):.4f}, Max: {np.max(stored_margins):.4f}")
        print(
            f"Margins < 0.05: {sum(1 for m in stored_margins if m < 0.05)} ({sum(1 for m in stored_margins if m < 0.05) / len(stored_margins) * 100:.1f}%)")
        print(
            f"Margins > 0.15: {sum(1 for m in stored_margins if m > 0.15)} ({sum(1 for m in stored_margins if m > 0.15) / len(stored_margins) * 100:.1f}%)")

    if stored_pos_sims:
        print(f"Stored pos_sims - Mean: {np.mean(stored_pos_sims):.4f}, Std: {np.std(stored_pos_sims):.4f}")

    if stored_neg_sims:
        print(f"Stored neg_sims - Mean: {np.mean(stored_neg_sims):.4f}, Std: {np.std(stored_neg_sims):.4f}")

    # Type distribution
    type_counts = defaultdict(int)
    for item in data:
        type_counts[item.get('type', 'unknown')] += 1
    print(f"\nTriplet types: {dict(type_counts)}")

    # Re-compute similarities with current model (sample only for speed)
    print("\n=== RE-COMPUTING WITH CURRENT MODEL ===")
    model = SentenceTransformer(model_name)

    # Sample for faster computation
    sample_size = min(500, len(data))
    sample_data = data[:sample_size]

    queries = [f"query: {item['question']}" for item in sample_data]
    positives = [f"passage: {item['positive']}" for item in sample_data]
    negatives = [f"passage: {item['negative']}" for item in sample_data]

    print(f"Encoding {sample_size} triplets...")
    query_embs = model.encode(queries, normalize_embeddings=True)
    pos_embs = model.encode(positives, normalize_embeddings=True)
    neg_embs = model.encode(negatives, normalize_embeddings=True)

    # Compute similarities
    pos_sims = [float(util.cos_sim(q, p).item()) for q, p in zip(query_embs, pos_embs)]
    neg_sims = [float(util.cos_sim(q, n).item()) for q, n in zip(query_embs, neg_embs)]
    computed_margins = [p - n for p, n in zip(pos_sims, neg_sims)]

    print(f"Re-computed margins - Mean: {np.mean(computed_margins):.4f}, Std: {np.std(computed_margins):.4f}")
    print(f"Re-computed margins - Min: {np.min(computed_margins):.4f}, Max: {np.max(computed_margins):.4f}")
    print(f"Re-computed pos_sims - Mean: {np.mean(pos_sims):.4f}, Std: {np.std(pos_sims):.4f}")
    print(f"Re-computed neg_sims - Mean: {np.mean(neg_sims):.4f}, Std: {np.std(neg_sims):.4f}")

    # Compare stored vs computed
    if len(stored_margins) >= sample_size:
        sample_stored_margins = stored_margins[:sample_size]
        margin_diff = np.array(computed_margins) - np.array(sample_stored_margins)
        print(
            f"\nStored vs Re-computed margin difference - Mean: {np.mean(margin_diff):.4f}, Std: {np.std(margin_diff):.4f}")

    # Distribution analysis
    print("\nRe-computed distribution:")
    print(
        f"Margins < 0.05: {sum(1 for m in computed_margins if m < 0.05)} ({sum(1 for m in computed_margins if m < 0.05) / len(computed_margins) * 100:.1f}%)")
    print(
        f"Margins 0.05-0.15: {sum(1 for m in computed_margins if 0.05 <= m <= 0.15)} ({sum(1 for m in computed_margins if 0.05 <= m <= 0.15) / len(computed_margins) * 100:.1f}%)")
    print(
        f"Margins > 0.15: {sum(1 for m in computed_margins if m > 0.15)} ({sum(1 for m in computed_margins if m > 0.15) / len(computed_margins) * 100:.1f}%)")

    # Problematic triplets
    print("\n=== PROBLEMATIC TRIPLETS ===")
    negative_margins = [m for m in computed_margins if m < 0]
    if negative_margins:
        print(f"CRITICAL: {len(negative_margins)} triplets have NEGATIVE margins! (pos_sim < neg_sim)")
        print(f"Negative margins: {negative_margins[:10]}")

    very_small_margins = [m for m in computed_margins if 0 <= m < 0.02]
    if very_small_margins:
        print(f"WARNING: {len(very_small_margins)} triplets have very small margins < 0.02")

    high_neg_sims = [n for n in neg_sims if n > 0.8]
    if high_neg_sims:
        print(f"WARNING: {len(high_neg_sims)} negatives have similarity > 0.8 (too similar to query)")

    low_pos_sims = [p for p in pos_sims if p < 0.3]
    if low_pos_sims:
        print(f"WARNING: {len(low_pos_sims)} positives have similarity < 0.3 (might be too dissimilar)")

    # Recommended training margin
    optimal_margin = np.percentile(computed_margins, 50)  # Median
    conservative_margin = np.percentile(computed_margins, 25)  # 25th percentile

    print("\n=== TRAINING RECOMMENDATIONS ===")
    print(f"Median margin (50th percentile): {optimal_margin:.4f}")
    print(f"Conservative margin (25th percentile): {conservative_margin:.4f}")
    print(f"RECOMMENDED: Use margin = {conservative_margin:.4f} for training")

    if conservative_margin > 0.15:
        print("WARNING: Your data has very high margins. Consider using a smaller dataset or different negatives.")
    elif conservative_margin < 0.03:
        print("WARNING: Your data has very small margins. The model may struggle to learn meaningful separations.")

    return {
        'stored_margins': stored_margins,
        'computed_margins': computed_margins,
        'pos_sims': pos_sims,
        'neg_sims': neg_sims,
        'recommended_margin': conservative_margin
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    args = parser.parse_args()
    results = analyze_dataset_margins(args.dataset, args.model)
    print(f"Recommended margin: {results['recommended_margin']:.4f}")


if __name__ == "__main__":
    main()
