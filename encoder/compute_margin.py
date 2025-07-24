import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import gc
import warnings

warnings.filterwarnings("ignore")


class GPUTripletMarginAnalyzer:
    def __init__(self, dataset_path: str, model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
                 batch_size: int = 32, max_length: int = 512, device: Optional[str] = None):
        """
        Initialize the analyzer with GPU acceleration and efficient processing.

        Args:
            dataset_path: Path to the JSON file containing triplets
            model_name: HuggingFace model for text embedding
            batch_size: Batch size for processing (adjust based on GPU memory)
            max_length: Maximum token length for text processing
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.triplets = []

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Enable mixed precision for faster computation
        self.use_fp16 = self.device.type == 'cuda'
        if self.use_fp16:
            self.model = self.model.half()
            print("Enabled FP16 for faster computation")

        print(f"Model loaded on {self.device}")

    def load_dataset(self) -> List[Dict]:
        """Load triplets from JSON file efficiently."""
        try:
            print(f"Loading dataset from {self.dataset_path}...")
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                self.triplets = data
            elif isinstance(data, dict) and 'triplets' in data:
                self.triplets = data['triplets']
            else:
                raise ValueError("Unexpected JSON structure. Expected list of triplets or dict with 'triplets' key.")

            print(f"Loaded {len(self.triplets)} triplets")
            return self.triplets

        except FileNotFoundError:
            print(f"Error: File {self.dataset_path} not found.")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {self.dataset_path}")
            return []

    def validate_triplets(self) -> bool:
        """Validate that all triplets have required fields."""
        required_fields = ['question', 'positive', 'negative']

        for i, triplet in enumerate(self.triplets):
            if not all(field in triplet for field in required_fields):
                print(f"Error: Triplet {i} missing required fields. Expected: {required_fields}")
                return False

            # Check if all values are strings
            for field in required_fields:
                if not isinstance(triplet[field], str):
                    print(f"Error: Triplet {i}, field '{field}' is not a string")
                    return False

        return True

    def mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_texts_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts in batches using GPU acceleration.

        Args:
            texts: List of texts to encode

        Returns:
            Tensor of normalized embeddings
        """
        all_embeddings = []

        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i + self.batch_size]

            # Tokenize batch
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        model_output = self.model(**encoded_input)
                else:
                    model_output = self.model(**encoded_input)

                # Apply mean pooling
                embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu())

            # Clear GPU cache periodically
            if i % (self.batch_size * 10) == 0:
                torch.cuda.empty_cache()

        # Concatenate all embeddings
        return torch.cat(all_embeddings, dim=0)

    def calculate_cosine_distances_gpu(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate cosine distances between anchor-positive and anchor-negative pairs using GPU.

        Returns:
            Tuple of (anchor_positive_distances, anchor_negative_distances)
        """
        if not self.triplets:
            return torch.tensor([]), torch.tensor([])

        print("Extracting unique texts...")
        # Extract all unique texts for efficient processing
        all_texts = set()
        for triplet in self.triplets:
            all_texts.update([triplet['question'], triplet['positive'], triplet['negative']])

        all_texts = list(all_texts)
        print(f"Found {len(all_texts)} unique texts")

        # Encode all texts
        print("Encoding all texts...")
        embeddings = self.encode_texts_batch(all_texts)

        # Create mapping from text to embedding index
        text_to_idx = {text: i for i, text in enumerate(all_texts)}

        print("Calculating cosine distances...")
        anchor_positive_distances = []
        anchor_negative_distances = []

        # Process triplets in batches for memory efficiency
        for i in tqdm(range(0, len(self.triplets), self.batch_size), desc="Processing triplets"):
            batch_triplets = self.triplets[i:i + self.batch_size]

            # Get indices for current batch
            anchor_indices = [text_to_idx[triplet['question']] for triplet in batch_triplets]
            positive_indices = [text_to_idx[triplet['positive']] for triplet in batch_triplets]
            negative_indices = [text_to_idx[triplet['negative']] for triplet in batch_triplets]

            # Get embeddings for current batch
            anchor_embs = embeddings[anchor_indices]
            positive_embs = embeddings[positive_indices]
            negative_embs = embeddings[negative_indices]

            # Calculate cosine similarities (embeddings are already normalized)
            ap_similarities = torch.sum(anchor_embs * positive_embs, dim=1)
            an_similarities = torch.sum(anchor_embs * negative_embs, dim=1)

            # Convert to distances (1 - similarity)
            ap_distances = 1 - ap_similarities
            an_distances = 1 - an_similarities

            anchor_positive_distances.extend(ap_distances.tolist())
            anchor_negative_distances.extend(an_distances.tolist())

        # Clear memory
        del embeddings
        torch.cuda.empty_cache()
        gc.collect()

        return torch.tensor(anchor_positive_distances), torch.tensor(anchor_negative_distances)

    def analyze_distances(self, ap_distances: torch.Tensor, an_distances: torch.Tensor) -> Dict:
        """
        Analyze the distance distributions and suggest optimal margin.

        Args:
            ap_distances: Anchor-positive distances
            an_distances: Anchor-negative distances

        Returns:
            Dictionary containing analysis results
        """
        # Convert to numpy for statistics
        ap_dist_np = ap_distances.numpy()
        an_dist_np = an_distances.numpy()

        # Calculate margins (negative distance - positive distance)
        margins = an_dist_np - ap_dist_np

        # Statistics
        stats = {
            'anchor_positive': {
                'mean': float(np.mean(ap_dist_np)),
                'median': float(np.median(ap_dist_np)),
                'std': float(np.std(ap_dist_np)),
                'min': float(np.min(ap_dist_np)),
                'max': float(np.max(ap_dist_np))
            },
            'anchor_negative': {
                'mean': float(np.mean(an_dist_np)),
                'median': float(np.median(an_dist_np)),
                'std': float(np.std(an_dist_np)),
                'min': float(np.min(an_dist_np)),
                'max': float(np.max(an_dist_np))
            },
            'margins': {
                'mean': float(np.mean(margins)),
                'median': float(np.median(margins)),
                'std': float(np.std(margins)),
                'min': float(np.min(margins)),
                'max': float(np.max(margins)),
                'percentile_25': float(np.percentile(margins, 25)),
                'percentile_75': float(np.percentile(margins, 75)),
                'percentile_10': float(np.percentile(margins, 10)),
                'percentile_90': float(np.percentile(margins, 90))
            }
        }

        # Suggest optimal margin with multiple strategies
        # Strategy 1: Conservative (10th percentile)
        conservative_margin = max(0.05, stats['margins']['percentile_10'])

        # Strategy 2: Moderate (25th percentile)
        moderate_margin = max(0.1, stats['margins']['percentile_25'])

        # Strategy 3: Aggressive (mean * 0.8)
        aggressive_margin = max(0.2, stats['margins']['mean'] * 0.8)

        stats['suggested_margins'] = {
            'conservative': conservative_margin,
            'moderate': moderate_margin,
            'aggressive': aggressive_margin,
            'recommended': moderate_margin  # Default recommendation
        }

        return stats

    def plot_distributions(self, ap_distances: torch.Tensor, an_distances: torch.Tensor, save_path: str = None):
        """
        Plot distance distributions and margin analysis.

        Args:
            ap_distances: Anchor-positive distances
            an_distances: Anchor-negative distances
            save_path: Optional path to save the plot
        """
        # Convert to numpy
        ap_dist_np = ap_distances.numpy()
        an_dist_np = an_distances.numpy()
        margins = an_dist_np - ap_dist_np

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Anchor-Positive distances
        axes[0, 0].hist(ap_dist_np, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Anchor-Positive Cosine Distances')
        axes[0, 0].set_xlabel('Cosine Distance')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(ap_dist_np), color='red', linestyle='--', label=f'Mean: {np.mean(ap_dist_np):.3f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Anchor-Negative distances
        axes[0, 1].hist(an_dist_np, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_title('Anchor-Negative Cosine Distances')
        axes[0, 1].set_xlabel('Cosine Distance')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(an_dist_np), color='blue', linestyle='--', label=f'Mean: {np.mean(an_dist_np):.3f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Overlaid distributions
        axes[1, 0].hist(ap_dist_np, bins=50, alpha=0.5, color='blue', label='Anchor-Positive', edgecolor='black')
        axes[1, 0].hist(an_dist_np, bins=50, alpha=0.5, color='red', label='Anchor-Negative', edgecolor='black')
        axes[1, 0].set_title('Distance Distributions Comparison')
        axes[1, 0].set_xlabel('Cosine Distance')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Margin distribution with percentiles
        axes[1, 1].hist(margins, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_title('Margin Distribution (AN_dist - AP_dist)')
        axes[1, 1].set_xlabel('Margin')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(margins), color='red', linestyle='--', label=f'Mean: {np.mean(margins):.3f}')
        axes[1, 1].axvline(np.percentile(margins, 25), color='orange', linestyle=':',
                           label=f'25th: {np.percentile(margins, 25):.3f}')
        axes[1, 1].axvline(np.percentile(margins, 10), color='purple', linestyle=':',
                           label=f'10th: {np.percentile(margins, 10):.3f}')
        axes[1, 1].axvline(0, color='black', linestyle='-', alpha=0.3, label='Zero margin')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def run_analysis(self, plot: bool = True, save_plot: str = None) -> Dict:
        """
        Run complete analysis on the triplet dataset with GPU acceleration.

        Args:
            plot: Whether to generate plots
            save_plot: Path to save the plot (optional)

        Returns:
            Dictionary containing analysis results
        """
        print("=" * 60)
        print("GPU-ACCELERATED TRIPLET MARGIN ANALYSIS")
        print("=" * 60)

        # Load and validate dataset
        self.load_dataset()
        if not self.triplets:
            print("No triplets loaded. Exiting.")
            return {}

        print("Validating triplets...")
        if not self.validate_triplets():
            print("Dataset validation failed. Exiting.")
            return {}

        # Calculate distances using GPU
        print("Calculating cosine distances using GPU...")
        ap_distances, an_distances = self.calculate_cosine_distances_gpu()

        if len(ap_distances) == 0 or len(an_distances) == 0:
            print("Failed to calculate distances. Exiting.")
            return {}

        # Analyze results
        print("Analyzing distance distributions...")
        stats = self.analyze_distances(ap_distances, an_distances)

        # Print comprehensive results
        self._print_results(stats)

        if plot:
            print("Generating plots...")
            self.plot_distributions(ap_distances, an_distances, save_plot)

        return stats

    def _print_results(self, stats: Dict):
        """Print comprehensive analysis results."""
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)

        print(f"\nDataset size: {len(self.triplets):,} triplets")
        print(f"Device used: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max length: {self.max_length}")

        print(f"\nAnchor-Positive distances:")
        print(f"  Mean: {stats['anchor_positive']['mean']:.4f}")
        print(f"  Median: {stats['anchor_positive']['median']:.4f}")
        print(f"  Std: {stats['anchor_positive']['std']:.4f}")
        print(f"  Range: [{stats['anchor_positive']['min']:.4f}, {stats['anchor_positive']['max']:.4f}]")

        print(f"\nAnchor-Negative distances:")
        print(f"  Mean: {stats['anchor_negative']['mean']:.4f}")
        print(f"  Median: {stats['anchor_negative']['median']:.4f}")
        print(f"  Std: {stats['anchor_negative']['std']:.4f}")
        print(f"  Range: [{stats['anchor_negative']['min']:.4f}, {stats['anchor_negative']['max']:.4f}]")

        print(f"\nMargin statistics (AN_dist - AP_dist):")
        print(f"  Mean: {stats['margins']['mean']:.4f}")
        print(f"  Median: {stats['margins']['median']:.4f}")
        print(f"  Std: {stats['margins']['std']:.4f}")
        print(f"  Range: [{stats['margins']['min']:.4f}, {stats['margins']['max']:.4f}]")
        print(f"  10th percentile: {stats['margins']['percentile_10']:.4f}")
        print(f"  25th percentile: {stats['margins']['percentile_25']:.4f}")
        print(f"  75th percentile: {stats['margins']['percentile_75']:.4f}")
        print(f"  90th percentile: {stats['margins']['percentile_90']:.4f}")

        print(f"\n🎯 MARGIN RECOMMENDATIONS:")
        print(f"  Conservative (10th percentile): {stats['suggested_margins']['conservative']:.4f}")
        print(f"  Moderate (25th percentile): {stats['suggested_margins']['moderate']:.4f}")
        print(f"  Aggressive (80% of mean): {stats['suggested_margins']['aggressive']:.4f}")
        print(f"  👉 RECOMMENDED: {stats['suggested_margins']['recommended']:.4f}")

        # Calculate insights
        ap_np = np.array([stats['anchor_positive']['mean']])
        an_np = np.array([stats['anchor_negative']['mean']])
        margins = an_np - ap_np

        negative_margin_count = sum(1 for m in margins if m < 0)
        hard_examples = sum(1 for i in range(len(self.triplets)) if
                            (stats['anchor_negative']['mean'] - stats['anchor_positive']['mean']) < 0)

        print(f"\n📊 INSIGHTS:")
        print(f"  • Average margin: {stats['margins']['mean']:.4f}")
        print(
            f"  • Separation quality: {'Good' if stats['margins']['mean'] > 0.2 else 'Moderate' if stats['margins']['mean'] > 0.1 else 'Poor'}")
        print(
            f"  • Hard examples ratio: {(abs(stats['margins']['min']) / (stats['margins']['max'] + 1e-8) * 100):.1f}%")

        print(f"\n💡 TRAINING RECOMMENDATIONS:")
        if stats['margins']['mean'] > 0.3:
            print("  • Your dataset has good separation - consider starting with moderate margin")
        elif stats['margins']['mean'] > 0.1:
            print("  • Moderate separation - use conservative margin and monitor training")
        else:
            print("  • Poor separation - use very small margin or consider data quality")


def main():
    """Main function to run the analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description='GPU-accelerated triplet dataset analysis for optimal TripletLoss margin')
    parser.add_argument('dataset_path', help='Path to the JSON file containing triplets')
    parser.add_argument('--model', default='paraphrase-mpnet-base-v2',
                        help='HuggingFace model name for embeddings')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing (adjust for GPU memory)')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum token length')
    parser.add_argument('--device', choices=['cuda', 'cpu'],
                        help='Device to use (auto-detect if not specified)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--save-plot', type=str, help='Path to save the plot')

    args = parser.parse_args()

    analyzer = GPUTripletMarginAnalyzer(
        dataset_path=args.dataset_path,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )

    analyzer.run_analysis(plot=not args.no_plot, save_plot=args.save_plot)


if __name__ == "__main__":
    analyzer = GPUTripletMarginAnalyzer(
        dataset_path="datasets/dataset_1.json",
        batch_size=16,
        max_length=512
    )
    results = analyzer.run_analysis()
