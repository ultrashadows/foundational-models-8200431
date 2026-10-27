import torch
import torch.nn.functional as F

from encoder.test.test import IREvaluator


def make_evaluator(block_size):
    evaluator = IREvaluator.__new__(IREvaluator)  # skip model loading
    evaluator.device = "cpu"
    evaluator.sim_block_docs = block_size
    return evaluator


def test_rank_and_topk_match_brute_force():
    torch.manual_seed(0)
    docs = F.normalize(torch.randn(50, 8), dim=1)
    query = F.normalize(torch.randn(8), dim=0)
    sims = docs @ query
    order = torch.argsort(sims, descending=True).tolist()

    for block_size in (7, 50, 64):
        evaluator = make_evaluator(block_size)
        for correct_idx in (order[0], order[10], order[-1]):
            top, scores, _, rank = evaluator._query_topk_and_rank(query, docs, 5, correct_idx)
            assert top == order[:5]
            assert rank == order.index(correct_idx) + 1
            assert scores == sorted(scores, reverse=True)
