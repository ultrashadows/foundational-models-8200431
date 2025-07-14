import json
import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    query_id: str
    question: str
    correct_doc_id: str
    retrieved_doc_ids: List[str]
    scores: List[float]
    correct_rank: Optional[int]


class IRMetrics:
    @staticmethod
    def hits_at_k(results: List[EvaluationResult], k: int) -> float:
        hits = sum(1 for r in results if r.correct_rank and r.correct_rank <= k)
        return hits / len(results)

    @staticmethod
    def hit_rate_at_k(results: List[EvaluationResult], k: int) -> float:
        return IRMetrics.hits_at_k(results, k)

    @staticmethod
    def precision_at_k(results: List[EvaluationResult], k: int) -> float:
        precisions = []
        for r in results:
            if r.correct_rank and r.correct_rank <= k:
                precisions.append(1.0 / k)
            else:
                precisions.append(0.0)
        return float(np.mean(precisions))

    @staticmethod
    def recall_at_k(results: List[EvaluationResult], k: int) -> float:
        return IRMetrics.hits_at_k(results, k)

    @staticmethod
    def f1_at_k(results: List[EvaluationResult], k: int) -> float:
        precision = IRMetrics.precision_at_k(results, k)
        recall = IRMetrics.recall_at_k(results, k)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def r_precision(results: List[EvaluationResult]) -> float:
        r_prec_scores = []
        for r in results:
            if r.correct_rank == 1:
                r_prec_scores.append(1.0)
            else:
                r_prec_scores.append(0.0)
        return float(np.mean(r_prec_scores))

    @staticmethod
    def rbp(results: List[EvaluationResult], p: float = 0.8) -> float:
        rbp_scores = []
        for r in results:
            if r.correct_rank:
                rbp_score = (1 - p) * (p ** (r.correct_rank - 1))
                rbp_scores.append(rbp_score)
            else:
                rbp_scores.append(0.0)
        return float(np.mean(rbp_scores))

    @staticmethod
    def mrr(results: List[EvaluationResult]) -> float:
        rr_scores = []
        for r in results:
            if r.correct_rank:
                rr_scores.append(1.0 / r.correct_rank)
            else:
                rr_scores.append(0.0)
        return float(np.mean(rr_scores))

    @staticmethod
    def map_score(results: List[EvaluationResult]) -> float:
        return IRMetrics.mrr(results)

    @staticmethod
    def dcg_at_k(results: List[EvaluationResult], k: int) -> float:
        dcg_scores = []
        for r in results:
            dcg = 0.0
            if r.correct_rank and r.correct_rank <= k:
                dcg = 1.0 / np.log2(r.correct_rank + 1)
            dcg_scores.append(dcg)
        return float(np.mean(dcg_scores))

    @staticmethod
    def ndcg_at_k(results: List[EvaluationResult], k: int) -> float:
        dcg = IRMetrics.dcg_at_k(results, k)
        idcg = 1.0
        return dcg / idcg if idcg > 0 else 0.0


class AlbertinaIREvaluator:
    def __init__(self,
                 model_name: str = "PORTULAN/albertina-1b5-portuguese-ptpt-encoder",
                 mongo_uri: str = "mongodb://localhost:27017/",
                 db_name: str = "database",
                 collection_name: str = "test",
                 cache_dir: str = "./cache",
                 max_length: int = 4096,
                 use_tokenizer: bool = True):

        self.model_name = model_name
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.use_tokenizer = use_tokenizer

        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        count = self.collection.count_documents({})
        logger.info(f"Number of documents in collection: {count}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)

        if self.use_tokenizer:
            logger.info(f"Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        os.makedirs(cache_dir, exist_ok=True)

        self.documents = None
        self.document_embeddings = None
        self.doc_id_to_index = {}

    def load_documents(self, force_reload: bool = False) -> None:
        cache_file = os.path.join(self.cache_dir, "documents.pkl")

        if not force_reload and os.path.exists(cache_file):
            logger.info("Loading documents from cache...")
            with open(cache_file, 'rb') as f:
                self.documents = pickle.load(f)
        else:
            logger.info("Loading documents from MongoDB...")
            cursor = self.collection.find({})
            self.documents = []

            for doc in tqdm(cursor, desc="Loading documents"):
                self.documents.append({
                    '_id': str(doc['_id']),
                    'Texto Integral': doc.get('Texto Integral', ''),
                    'results': doc.get('results', [])
                })

            with open(cache_file, 'wb') as f:
                pickle.dump(self.documents, f)

        self.doc_id_to_index = {doc['_id']: i for i, doc in enumerate(self.documents)}
        logger.info(f"Loaded {len(self.documents)} documents")

    def preprocess_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        text = ' '.join(text.split())

        if self.use_tokenizer:
            tokens = self.tokenizer.tokenize(text)

            if len(tokens) > self.max_length - 2:
                tokens = tokens[:self.max_length - 2]
                text = self.tokenizer.convert_tokens_to_string(tokens)

        return text

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        processed_texts = [self.preprocess_text(text) for text in texts]

        embeddings = []
        for i in tqdm(range(0, len(processed_texts), batch_size), desc="Encoding texts"):
            batch = processed_texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=True,
                device=self.device,
                truncation=True,
            )
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def compute_document_embeddings(self, force_recompute: bool = False) -> None:
        if self.documents is None:
            self.load_documents()

        cache_file = os.path.join(self.cache_dir, "document_embeddings.pkl")

        if not force_recompute and os.path.exists(cache_file):
            logger.info("Loading document embeddings from cache...")
            with open(cache_file, 'rb') as f:
                self.document_embeddings = pickle.load(f)
        else:
            logger.info("Computing document embeddings with tokenization...")
            texts = [doc['Texto Integral'] for doc in self.documents]

            self.document_embeddings = self.encode_texts(texts, batch_size=32)

            with open(cache_file, 'wb') as f:
                pickle.dump(self.document_embeddings, f)

        logger.info(f"Document embeddings shape: {self.document_embeddings.shape}")

    def extract_questions(self) -> List[Tuple[str, str, str]]:
        questions = []

        for doc in self.documents:
            doc_id = doc['_id']
            results = doc.get('results', [])

            for result in results:
                if 'pergunta' in result:
                    questions.append((
                        f"{doc_id}_{len(questions)}",
                        result['pergunta'],
                        doc_id
                    ))

        logger.info(f"Extracted {len(questions)} questions")
        return questions

    def retrieve_documents(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        query_embedding = self.encode_texts([query], batch_size=1)[0]

        similarities = np.dot(self.document_embeddings, query_embedding)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        doc_ids = [self.documents[i]['_id'] for i in top_indices]
        scores = [similarities[i] for i in top_indices]

        return doc_ids, scores

    def evaluate_single_query(self, question: str, correct_doc_id: str, top_k: int = 5) -> EvaluationResult:
        retrieved_docs, scores = self.retrieve_documents(question, top_k)

        correct_rank = None
        if correct_doc_id in retrieved_docs:
            correct_rank = retrieved_docs.index(correct_doc_id) + 1  # 1-based rank

        return EvaluationResult(
            query_id=f"q_{len(retrieved_docs)}",
            question=question,
            correct_doc_id=correct_doc_id,
            retrieved_doc_ids=retrieved_docs,
            scores=scores,
            correct_rank=correct_rank
        )

    def run_evaluation(self, max_questions: Optional[int] = None, top_k: int = 5) -> Dict:
        logger.info("Starting evaluation...")

        self.load_documents()
        self.compute_document_embeddings()

        questions = self.extract_questions()

        if max_questions:
            questions = questions[:max_questions]

        results = []
        for question_id, question, correct_doc_id in tqdm(questions, desc="Evaluating questions"):
            result = self.evaluate_single_query(question, correct_doc_id, top_k)
            result.query_id = question_id
            results.append(result)

        metrics = self.calculate_all_metrics(results)

        metrics['evaluation_info'] = {
            'total_questions': len(results),
            'total_documents': len(self.documents),
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'top_k': top_k
        }

        return metrics

    def calculate_all_metrics(self, results: List[EvaluationResult]) -> Dict:
        metrics = {}

        k_values = [1, 3, 5]

        for k in k_values:
            metrics[f'hits@{k}'] = IRMetrics.hits_at_k(results, k)
            metrics[f'hit_rate@{k}'] = IRMetrics.hit_rate_at_k(results, k)

        for k in k_values:
            metrics[f'precision@{k}'] = IRMetrics.precision_at_k(results, k)
            metrics[f'recall@{k}'] = IRMetrics.recall_at_k(results, k)
            metrics[f'f1@{k}'] = IRMetrics.f1_at_k(results, k)

        metrics['r_precision'] = IRMetrics.r_precision(results)

        metrics['rbp'] = IRMetrics.rbp(results)

        metrics['mrr'] = IRMetrics.mrr(results)

        metrics['map'] = IRMetrics.map_score(results)

        for k in k_values:
            metrics[f'dcg@{k}'] = IRMetrics.dcg_at_k(results, k)
            metrics[f'ndcg@{k}'] = IRMetrics.ndcg_at_k(results, k)

        return metrics

    def save_results(self, metrics: Dict, filename: str = None) -> None:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {filename}")

    def print_results(self, metrics: Dict) -> None:
        print("\n" + "=" * 60)
        print("INFORMATION RETRIEVAL EVALUATION RESULTS")
        print("=" * 60)

        print(f"\nModel: {metrics['evaluation_info']['model_name']}")
        print(f"Total Questions: {metrics['evaluation_info']['total_questions']}")
        print(f"Total Documents: {metrics['evaluation_info']['total_documents']}")
        print(f"Evaluation Time: {metrics['evaluation_info']['timestamp']}")

        print("\n" + "-" * 40)
        print("RANKING METRICS")
        print("-" * 40)

        print(f"MRR (Mean Reciprocal Rank): {metrics['mrr']:.4f}")
        print(f"MAP (Mean Average Precision): {metrics['map']:.4f}")
        print(f"R-Precision: {metrics['r_precision']:.4f}")
        print(f"RBP (Rank-Biased Precision): {metrics['rbp']:.4f}")

        print("\n" + "-" * 40)
        print("HITS AND PRECISION METRICS")
        print("-" * 40)

        for k in [1, 3, 5]:
            print(f"Hits@{k}: {metrics[f'hits@{k}']:.4f}")
            print(f"Precision@{k}: {metrics[f'precision@{k}']:.4f}")
            print(f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
            print()


def main():
    evaluator = AlbertinaIREvaluator()

    metrics = evaluator.run_evaluation()

    evaluator.print_results(metrics)
    evaluator.save_results(metrics)


if __name__ == "__main__":
    main()
