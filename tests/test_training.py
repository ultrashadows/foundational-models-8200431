import json

import torch
import torch.nn.functional as F

from encoder.train.dataset_processor import DatasetProcessor
from encoder.train.triplet_loss import TripletLoss


def unit(*values):
    return F.normalize(torch.tensor([values], dtype=torch.float32), dim=-1)


def test_triplet_loss_is_zero_when_negative_is_far():
    loss = TripletLoss(base_margin=0.06)
    anchor, positive, negative = unit(1.0, 0.0), unit(1.0, 0.0), unit(-1.0, 0.0)
    assert loss(anchor, positive, negative).item() == 0.0


def test_triplet_loss_penalises_negative_closer_than_positive():
    loss = TripletLoss(base_margin=0.06, reduction="none")
    anchor, positive, negative = unit(1.0, 0.0), unit(0.0, 1.0), unit(1.0, 0.1)
    value = loss(anchor, positive, negative)
    assert value.item() > 0.0
    weighted = loss(anchor, positive, negative, sample_weights=torch.tensor([0.5]))
    assert torch.isclose(weighted, value * 0.5)


def write_dataset(tmp_path, n=20):
    samples = [{"question": f"q {i}", "positive": f"pos {i}", "negative": f"neg {i}"} for i in range(n)]
    path = tmp_path / "triplets.json"
    path.write_text(json.dumps(samples), encoding="utf-8")
    return path


def test_split_is_90_10_and_disjoint(tmp_path, tokenizer):
    path = write_dataset(tmp_path)
    train = DatasetProcessor(tokenizer, "train", 8, path)
    test = DatasetProcessor(tokenizer, "test", 8, path)
    assert (len(train), len(test)) == (18, 2)
    assert not {s["question"] for s in train.samples} & {s["question"] for s in test.samples}


def test_train_order_is_identical_across_instances(tmp_path, tokenizer):
    # Every DDP rank builds its own dataset. DistributedSampler needs them to agree on the order.
    path = write_dataset(tmp_path)
    orders = [[s["question"] for s in DatasetProcessor(tokenizer, "train", 8, path).samples] for _ in range(3)]
    assert orders[0] == orders[1] == orders[2]


def test_negatives_come_from_random_pool(tmp_path, tokenizer, monkeypatch):
    # Documents the thesis behaviour: even rand just below 1.0 takes the random-pool branch.
    import encoder.train.dataset_processor as dp

    dataset = DatasetProcessor(tokenizer, "train", 8, write_dataset(tmp_path))
    picked = []
    monkeypatch.setattr(dp.random, "random", lambda: 0.999)
    monkeypatch.setattr(dp.random, "choice", lambda pool: picked.append(pool) or "random text")
    dataset[0]
    assert picked and picked[0] is dataset.negative_pool


def test_getitem_returns_fixed_length_triplet(tmp_path, tokenizer):
    item = DatasetProcessor(tokenizer, "train", 8, write_dataset(tmp_path))[0]
    for part in ("anchor", "positive", "negative"):
        assert item[part]["input_ids"].shape == (8,)
    assert item["weight"].item() == 1.0
