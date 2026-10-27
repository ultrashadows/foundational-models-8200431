from crawler.dgsi.external import stj
from crawler.dgsi.internal import HEADER_BGCOLOR, parse_document
from encoder.test.clear import select_middle
from encoder.train.cache_processor import CacheProcessor
from helpers.embed import fill_embeddings
from tests.conftest import WhitespaceTokenizer


def test_dgsi_parser_skips_section_headers():
    html = f"""<table>
      <tr><td>Processo:</td><td> 123/45 </td></tr>
      <tr><td>Secção</td><td bgcolor="{HEADER_BGCOLOR}">header</td></tr>
      <tr><td>Decisão Texto Integral:</td><td>Acordam os juízes</td></tr>
    </table>"""
    assert parse_document(html) == {"Processo": "123/45", "Decisão Texto Integral": "Acordam os juízes"}


def test_stj_parser_does_not_reuse_previous_value():
    html = """<div class="container-fluid">
      <div class="row border-bottom"><div class="col-1">Processo:</div><div class="col-7">1/23</div></div>
      <div class="row border-bottom"><div class="col-1">Relator:</div></div>
    </div>"""
    assert stj.parse_document(html) == {"Processo": "1/23"}


def make_cache_processor(**kwargs):
    processor = CacheProcessor(tokenizer=WhitespaceTokenizer(), max_length=30, chunk_overlap=5,
                               use_chunking_threshold=10, min_chunk_tokens=3, enable_cache=False, **kwargs)
    processor._create_worker_tokenizer = WhitespaceTokenizer
    return processor


def test_chunking_covers_document_with_overlap():
    processor = make_cache_processor()
    words = [f"w{i}" for i in range(25)]
    chunks = processor._tokenize_and_chunk_fast(" ".join(words), WhitespaceTokenizer())
    # effective length 10, stride 5
    assert chunks[0].split() == words[:10]
    assert chunks[1].split() == words[5:15]
    assert chunks[-1].split()[-1] == "w24"


def test_short_document_is_not_chunked():
    result = make_cache_processor()._process_positive_only({"Texto Integral": "a b c"}, WhitespaceTokenizer())
    assert result == {"correct_chunks": ["a b c"], "is_chunked": False, "correct_doc_text": "a b c"}


def test_document_batch_uses_worker_tokenizer():
    processor = make_cache_processor()
    processor.tokenizer = None  # the shared tokenizer must not be touched by workers
    docs = [{"_id": {"$oid": str(i)}, "Texto Integral": " ".join(["x"] * 40)} for i in range(3)]
    results = processor._process_document_batch(docs)
    assert all(r and r["is_chunked"] for r in results)
    assert processor.get_processing_stats()["errors"] == 0


class FakeModel:
    def encode(self, texts):
        import numpy as np
        return np.array([[float(len(t)), 1.0] for t in texts])


class FakeCollection:
    def __init__(self):
        self.updates = []

    def update_one(self, query, update):
        self.updates.append(query["_id"])


def test_fill_embeddings_keeps_ids_aligned():
    docs = [
        {"_id": "a", "Texto Integral": "aaa"},
        {"_id": "b", "embedding": [9.0, 9.0]},
        {"_id": "c"},  # no text, must be skipped without shifting later ids
        {"_id": "d", "Texto Integral": "d"},
    ]
    embeddings, ids = fill_embeddings(FakeModel(), docs, FakeCollection())
    assert ids == ["b", "a", "d"]
    assert embeddings == [[9.0, 9.0], [3.0, 1.0], [1.0, 1.0]]
    assert docs[0]["embedding"] == [3.0, 1.0]


def test_select_middle_keeps_original_order():
    data = [{"Texto Integral": "x" * n} for n in (5, 1, 4, 2, 3)]
    assert select_middle(data, 1, 4) == [data[2], data[3], data[4]]
