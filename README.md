# Portuguese Legal Retrieval Encoder

Code for building a corpus of Portuguese court decisions, generating exam-style questions about
them with an LLM, mining (question, positive, negative) triplets, and fine-tuning the Albertina
1.5B Portuguese encoder with a triplet loss so that a question retrieves the decision it was
written from. It also contains the script that measures that retrieval quality (recall@k and MRR).

## Overview

The research question is whether a general Portuguese encoder can be adapted, with synthetic
supervision only, to retrieve the right court decision for a legal question.

The approach implemented here:

1. Crawl public jurisprudence databases (DGSI, STJ, CSM and PGR) into MongoDB.
2. Ask `gpt-4o-mini` to write five conceptual exam questions (with answers) for sampled Supreme
   Court (STJ) decisions. These questions become the queries.
3. Merge and clean the whole crawl into one pool of decisions, used as the source of negatives.
4. For each question, take the most similar chunk of its own decision as the positive and a pool
   chunk with a controlled similarity gap as the negative.
5. Fine-tune `PORTULAN/albertina-1b5-portuguese-ptpt-encoder` on those triplets with an
   adaptive-margin triplet loss using multi-GPU DDP.
6. Evaluate question-to-decision retrieval on a held-out set of decisions.

The repository holds code and the generation prompt only. The crawled corpus, the exported JSON
files, the triplet dataset and the model weights are not included (see [Data](#data-requirements)
and [Models](#models-and-checkpoints)).

## Project status

This is the implementation behind a completed Master's thesis. After the thesis was accepted the
code was cleaned up for archival and publication. The cleanup fixed bugs, removed dead and
duplicated code, made paths and settings configurable and added tests and documentation. It did
not change the research method. Changes that can affect results are listed under
[Changes made during the archival cleanup](#changes-made-during-the-archival-cleanup).

## Capabilities

- Crawl 15 DGSI databases plus the STJ, CSM and PGR portals into MongoDB.
- Submit OpenAI Batch API jobs that generate questions for STJ decisions and store the answers.
- Merge, normalise and length-filter the crawl into a single negative pool.
- Chunk documents with the Albertina tokenizer and cache the chunks.
- Mine 125,000 triplets with margin-controlled negatives and per-chunk usage limits.
- Fine-tune an encoder with DDP (bf16 autocast, gradient accumulation, TensorBoard logging).
- Evaluate retrieval with recall@1/3/5/10/20 and MRR, plus a failure-case printout.

## Repository structure

```text
.
├── db.py                          MongoDB connection (reads MONGO_URI)
├── crawler/dgsi/
│   ├── helpers.py                 DGSI listing-page parser
│   ├── internal.py                crawler for the 15 DGSI databases (one court per run)
│   └── external/
│       ├── csm.py                 jurisprudencia.csm.org.pt
│       ├── pgr.py                 ministeriopublico.pt/pareceres-pgr
│       └── stj.py                 juris.stj.pt
├── processing/                    files.* collections to database.pool
│   ├── aggregator.py              merge the court collections
│   ├── standardize.py             keep one full-text field per document
│   └── prepare.py                 drop documents under 50 words
├── question_generation/           OpenAI batch question generation for files.stj
│   ├── prompt.txt                 exact prompt used for the thesis
│   ├── create_request.py
│   ├── request_status.py
│   └── retrieve_batch.py
├── helpers/
│   ├── triplet_generator.py       triplet mining
│   ├── embed.py                   "most distant document" (wrong field, legacy)
│   ├── find_distant.py            store embeddings in database.encoder (legacy)
│   └── delete_wrong.py            clear the wrong field (legacy)
├── encoder/
│   ├── train/
│   │   ├── database.py            build database.encoder
│   │   ├── cache_processor.py     chunk cache for dataset.json
│   │   ├── compute_margin.py      margin statistics of a triplet file
│   │   ├── dataset_processor.py   PyTorch dataset of triplets
│   │   ├── triplet_loss.py        adaptive-margin triplet loss
│   │   ├── wrapper.py             CLS encoder used only by the disabled hard-negative option
│   │   └── fine_tune.py           DDP training
│   └── test/
│       ├── clear.py               select the evaluation subset by document length
│       └── test.py                retrieval evaluation
├── tests/                         unit tests (CPU only, no data needed)
├── .github/workflows/checks.yml   CI: ruff and pytest on CPU
├── .pre-commit-config.yaml        local hooks: gitleaks, ruff, large-file and private-key checks
├── .env.example                   environment variables, copy to .env
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml                 ruff and pytest settings
├── CITATION.cff
├── LICENSE                        MIT
└── SECURITY.md
```

`encoder/` is also the default working directory for data and artifacts. The files the
pipelines read and write there (`dataset.json`, `pool.json`, `cache/`, `datasets/`, `models/`,
`results/`, `runs/`) are ignored by git.

## System requirements

- Python 3.12 (the version used during the thesis).
- MongoDB, reachable from the machine running the crawl, generation and processing steps.
- An OpenAI API key for question generation.
- NVIDIA GPUs with CUDA for fine-tuning and evaluation. Training uses the NCCL backend and bf16
  autocast, so it needs GPUs with bfloat16 support. The code was developed on Windows and trained
  on a GPU cluster (a commit message mentions the Deucalion supercomputer). The number and type of
  GPUs used are not recorded in the repository.
- Memory. The triplet miner keeps up to 2,000,000 pool chunk embeddings on the GPU in fp16
  (about 3 GB for 768-dimensional vectors) plus their texts in RAM. A code comment describes
  `pool.json` as about 7 GB. Albertina 1.5B with batch size 12 and 512 tokens per text needs a
  large-memory GPU, even with gradient checkpointing.
- Only the unit tests have been run on CPU. No pipeline has been tested on Linux or macOS as part
  of the cleanup, although nothing in the code is Windows-specific.

## Installation

```bash
git clone <repository-url>
cd foundational-models-8200431
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

For tests and linting also run `pip install -r requirements-dev.txt`. To check commits for
secrets and large files before they are made, run `pip install pre-commit` and `pre-commit install`.

Copy `.env.example` to `.env` and fill in the values you need.

All commands below are run from the repository root with `python -m <module>`. The modules import
each other by their full path (for example `from db import connect`), which only works when the
repository root is on the Python path.

## Configuration

| Setting | Where | Default |
|---|---|---|
| `MONGO_URI` | environment variable, read in `db.py` | `mongodb://localhost:27017` |
| `OPENAI_API_KEY` | environment variable or `.env` file | none |
| `OPENAI_PROMPT` | optional environment variable that overrides `question_generation/prompt.txt` | unset |
| Training settings | `python -m encoder.train.fine_tune --help` | values of the final thesis run |
| Evaluation settings | `python -m encoder.test.test --help` | see below |
| Mining constants | top of `helpers/triplet_generator.py`, some exposed as flags | see below |

MongoDB layout used by the code:

| Database.collection | Written by | Contents |
|---|---|---|
| `files.<court>` | crawlers | one collection per source (`jstj`, `jtrl`, `stj`, `csm`, `pgr` and so on) |
| `files.stj` (+ `results`, `wrong`) | `retrieve_batch` | STJ decisions with generated questions |
| `database.pool` | `processing/*` | merged and cleaned corpus (`_id`, `Texto Integral`) |
| `database.encoder` | `encoder.train.database` | decisions with questions, used to build `dataset.json` |

## Data requirements

All data come from public Portuguese jurisprudence websites. Nothing is redistributed here.
Rerunning the crawlers produces a newer snapshot than the one used for the thesis, since the
sites keep adding decisions. Check each site's terms of use before redistributing the crawl or
any data derived from it. Court decisions published there are pseudonymised by the courts
(parties appear as AA, BB), but they still describe real cases.

Files the training pipeline expects under `encoder/`:

| File | Produced by | Format |
|---|---|---|
| `dataset.json` | export of `database.encoder` | JSON array in MongoDB Extended JSON (`"_id": {"$oid": ...}`), with `Texto Integral` and `results` |
| `pool.json` | export of `database.pool` | JSON array of documents with `Texto Integral` |
| `cache/<oid>.pkl` | `encoder.train.cache_processor` | chunk cache, one file per document |
| `datasets/dataset_1.json` | `helpers.triplet_generator` | list of triplets |
| `test/evaluation/test.json` | not in the repository | evaluation documents, same format as `dataset.json` |
| `test/evaluation/clean.json` | `encoder.test.clear` | subset of `test.json` |

The two exports are not scripted in the repository. The format the loaders expect matches
`mongoexport --jsonArray`, for example:

```bash
mongoexport --uri "$MONGO_URI" --db database --collection encoder --jsonArray --out encoder/dataset.json
mongoexport --uri "$MONGO_URI" --db database --collection pool --jsonArray --out encoder/pool.json
```

The origin of `encoder/test/evaluation/test.json` is not recorded. It has the same structure as
`dataset.json`, with generated questions, so it was most likely exported from the same
collections. Whether its decisions are disjoint from those in `dataset.json` cannot be checked
from the code (see [Known limitations](#known-limitations)).

The triplet miner also accepts a sharded pool cache (`encoder/pool_cache/pool_index.json` with
`pool_embeds_NNNNN.pt` and `pool_texts_NNNNN.pkl`). It is loaded before anything else if present.
The script that produced those shards is not in the repository.

## Models and checkpoints

| Model | Role | Source |
|---|---|---|
| `PORTULAN/albertina-1b5-portuguese-ptpt-encoder` | model that is fine-tuned, and tokenizer used for chunking | Hugging Face |
| `intfloat/multilingual-e5-base` | embedding model for triplet mining and `compute_margin` | Hugging Face |
| `paraphrase-mpnet-base-v2` | embeddings for the legacy `wrong` field (`helpers/embed.py`) | Hugging Face (sentence-transformers) |
| `paraphrase-multilingual-mpnet-base-v2` | embeddings stored by `helpers/find_distant.py` | Hugging Face (sentence-transformers) |
| `gpt-4o-mini` | question generation | OpenAI API |

Fine-tuned weights are not included. Expected locations:

- `encoder/models/albertina-1b5/` is a local copy of the base model, used as the starting point of
  the first training stage. It is presumably the PORTULAN model above.
- `encoder/models/finetuned-albertina-1b5-100k-triplets/` is the starting point of the second stage
  and the default model for evaluation.
- `encoder/results/finetuned-albertina-1b5-100k-triplets/` is where training writes the final
  model, together with `training_args.json`.
- `encoder/results/checkpoint_epoch_N/` holds checkpoints of every epoch except the last.

Training writes to `results/` and reads from `models/`. Moving a trained model from one to the
other is a manual step.

## Pipelines

Every pipeline below lists its entry point, inputs and outputs. Steps 1 to 4 need MongoDB.

### 1. Crawling

**Purpose.** Download decisions with their metadata and full text.

**Entry points.**

```bash
python -m crawler.dgsi.internal <court>     # court is one of: atco1 cajp jcon jdgpj jsta jstj jtca
                                            # jtcampca jtcampct jtcn jtrc jtre jtrg jtrl jtrp
python -m crawler.dgsi.external.stj [--start-page N]   # default 0
python -m crawler.dgsi.external.csm [--start-page N]   # default 1
python -m crawler.dgsi.external.pgr [--start-page N]   # default 0
```

**Outputs.** One document per decision in `files.<court>`. Field names are those of each website
(for example `Processo`, `Sumário`, `Texto Integral`, `Decisão Texto Integral`).

**Notes.** The DGSI crawler follows the "Seguinte" link until the last page. The external
crawlers retry a page whose documents all failed and stop at the first empty page. The original
scripts started at hard-coded resume points (STJ page 6889, CSM page 2263, PGR page 1), which are
now `--start-page`. The external crawlers turn off TLS certificate checks because the sites'
certificates did not validate. Crawls are not resumable beyond `--start-page` and do not
deduplicate, so rerunning a crawl inserts duplicates.

### 2. Question generation

**Purpose.** Create the queries. For each sampled STJ decision the model gets the prompt in
`question_generation/prompt.txt` followed directly by the decision text, and returns a JSON list
of five `pergunta`/`resposta` pairs.

**Prerequisites.** `files.stj` from step 1 and `OPENAI_API_KEY`.

**Commands.**

```bash
python -m question_generation.create_request [--sample-size 50]   # prints a batch id
python -m question_generation.request_status <batch_id>          # repeat until it has an output_file_id
python -m question_generation.retrieve_batch <output_file_id>
```

**Outputs.** `create_request` writes the batch input to `question_generation/requests.jsonl` and
submits it. It samples documents that do not have `results` yet, so running it repeatedly covers
more decisions. `retrieve_batch` stores `results` and `wrong` (the most distant document, see
step 4) in each answered document. Answers that are not valid JSON are skipped.

### 3. Pool construction

**Purpose.** Build the corpus that negatives are drawn from.

**Commands** (in this order):

```bash
python -m processing.aggregator     # clears database.pool and copies 16 collections into it
python -m processing.standardize    # keeps _id and one Texto Integral field
python -m processing.prepare        # drops documents with fewer than 50 words
```

**Details.** `aggregator` copies `cajp csm jcon jsta jstj jtca jtcampca jtcampct jtcn jtrc jtre
jtrg jtrl jtrp pgr stj` and adds `_origin_collection`. `atco1` and `jdgpj` are crawled but not
included. `standardize` treats every field whose name contains "integral" (any case) as the full
text and keeps the longest one. Documents without such a field are removed. `aggregator` writes
`migration.log` in the current directory. Export the result to `encoder/pool.json` afterwards.

### 4. Encoder dataset collection

**Purpose.** Gather the decisions that have questions.

```bash
python -m encoder.train.database add           # replaces database.encoder with files.jsta + files.stj docs that have results and wrong
python -m encoder.train.database standardize   # normalises result keys to pergunta/resposta
```

Export the result to `encoder/dataset.json` afterwards.

**Legacy negative selection.** `helpers/find_distant.py`, `helpers/delete_wrong.py` and
`helpers/embed.py` compute embeddings for `database.encoder`, clear `wrong`, and recompute `wrong`
as the least similar document. The final triplets do not use `wrong`. The field only matters
because `database add` requires it. The run order in the thesis is not recorded. Each script is
run with `python -m helpers.<name>`.

### 5. Chunk cache

**Purpose.** Split every decision into chunks of up to 492 Albertina tokens with 128 tokens of
overlap (chunks under 150 tokens are dropped). Decisions of up to 350 tokens are kept whole.

```bash
python -m encoder.train.cache_processor [--dataset encoder/dataset.json] [--cache-dir encoder/cache] [--force-refresh]
```

**Output.** `encoder/cache/<oid>.pkl`. Already cached documents are skipped.

### 6. Triplet mining

**Purpose.** Produce the training triplets.

**Prerequisites.** `encoder/dataset.json`, `encoder/cache/` and either `encoder/pool.json` or a
pool cache in `encoder/pool_cache/`.

```bash
python -m helpers.triplet_generator [--model intfloat/multilingual-e5-base] [--target-triplets 125000]
```

Other flags: `--embedding-batch-size`, `--query-batch-size`, `--max-chunk-usage` (3),
`--chunk-size` (512), `--chunk-overlap` (128), `--checkpoint-every` (5000), `--memory-aggressive`,
`--rebuild-pool`, `--clear-checkpoints`, `--debug`.

**Method.** Questions are encoded as `query: <question>` and chunks of their own decision as
`passage: <chunk>` (the E5 convention). The positive is the most similar own chunk. A pool chunk
is a candidate negative if its similarity to the question is between 0.20 and
`min(0.88 * pos_sim, 0.88)`. Among candidates the miner prefers a margin
(`pos_sim - neg_sim`) within 0.20 ± 0.08, then a larger margin, then a smaller one of at least
0.12, with fallbacks when no candidate exists. Each pool chunk can be used three times. When more
than the target number of triplets is produced, the ones with margin closest to 0.20 are kept.
Seeds are fixed to 42.

**Outputs.** `encoder/datasets/dataset_1.json` with `doc_id`, `question`, `positive`, `negative`,
`pos_sim`, `neg_sim`, `margin`, `type` (semi-hard, hard, easy or a fallback label) and
`pool_chunk_idx`. Checkpoints go to `encoder/triplet_checkpoints/` and a resumed run continues
from the latest one. Also writes `triplet_mining.log` and caches the encoded pool in
`encoder/pool_cache/`.

**Analysis.** `python -m encoder.train.compute_margin [--dataset ...]` prints the stored margin
statistics and recomputes them on the first 500 triplets.

### 7. Fine-tuning

**Purpose.** Train the encoder.

```bash
torchrun --nproc_per_node=<num_gpus> -m encoder.train.fine_tune \
    --model encoder/models/albertina-1b5 --epochs 3 --lr 5e-6     # stage 1 (from history)
torchrun --nproc_per_node=<num_gpus> -m encoder.train.fine_tune   # stage 2 (defaults)
```

`torchrun` is needed even on one GPU.

**Defaults** (the last training run in the git history): start from
`encoder/models/finetuned-albertina-1b5-100k-triplets`, dataset `encoder/datasets/dataset_1.json`,
2 epochs, learning rate 2e-6, AdamW with weight decay 0.01, per-GPU batch 12 with 6 accumulation
steps, sequences of 512 tokens, linear schedule with `max(200, 10%)` warmup steps, gradient
clipping at 1.0, seed 42.

**Two stages.** The previous version of the script (git history, commit `4ea6de4`) trained from
`./models/albertina-1b5` for 3 epochs at 5e-6. The final commit changed the start point to the
model produced by that run, and set 2 epochs at 2e-6. The stage 1 command above reproduces those
settings. Both stages read the same `dataset_1.json`.

**Method.**

- Every text is mean-pooled over its tokens and L2-normalised.
- The loss is `relu(d_pos - d_neg + m)` with cosine distance and an adaptive margin
  `m = base_margin * sigmoid(10 * (d_neg - d_pos))`.
- Before training, rank 0 measures `pos_sim - neg_sim` on about 1,000 training triplets and sets
  `base_margin = max(0.03, min(0.7 * p25, 0.5 * median))`.
- The first 90% of the triplet file is used for training. The remaining 10% is not used anywhere.
- The negative of every training sample is a random text from the train split (the positive or
  negative of any sample, which can include the sample's own positive). See
  [Known limitations](#known-limitations) for why the mined negative is never used.
- Light text augmentation is applied with probability 0.03 to questions and positives and 0.02 to
  negatives: word dropout, one word swap, or legal synonym substitution.
- Sample weights are passed to the loss but are always 1.0, because the hard-negative
  precomputation that would set them (`hard_negatives=True`) is disabled.
- Early stopping on the training loss with patience 2 cannot trigger in a 2-epoch run.

**Outputs.**

- The final model and `training_args.json` in `--output-dir`.
- Checkpoints of each epoch except the last in `--checkpoint-dir`.
- TensorBoard logs in `--runs-dir`.
- `training.log` in the current directory.

### 8. Evaluation

**Purpose.** Measure question-to-decision retrieval.

```bash
python -m encoder.test.clear [--input encoder/test/evaluation/test.json] [--start 500 --end 1500]
python -m encoder.test.test [--model ...] [--eval-data encoder/test/evaluation/clean.json] [--output ...]
```

**Method.**

- `clear` sorts the evaluation documents by text length and keeps ranks `start` to `end`, which
  drops the shortest and longest decisions.
- `test` uses every generated question with more than 5 characters as a query. Its source
  decision is the only relevant document, and duplicate decision texts are counted once.
- Documents longer than 512 tokens are encoded as overlapping 512-token windows (overlap 128,
  stride adapted to at most 400 windows), mean-pooled, averaged and re-normalised.
- Queries are encoded the same way without windows. Scoring uses cosine similarity in fp16.
- recall@k is the share of queries whose decision is in the top k. MRR is the mean of
  1 / rank of that decision.

**Outputs.**

- The metrics are printed, compared with base-model reference values that are hard-coded in the
  script, and written to `encoder/test/finetuned_evaluation_results.json`.
- When recall@5 is below 0.5, five failure cases are printed.
- Document embeddings are cached under `encoder/cache/embeddings/<model>-<hash>/`.

The run that produced the hard-coded base-model values is not recorded. The original script
selected ranks 350 to 650 into `clean.json` and the final version selected 500 to 1500, but wrote
them to another project. Which subset produced the thesis numbers cannot be told from the code.

## End-to-end workflow

```text
crawl (1) ──> files.* ──> pool construction (3) ──> database.pool ──> pool.json ─────────┐
                │                                                                       │
                └─ files.stj ──> question generation (2) ──> encoder dataset (4)        │
                                                             └─> dataset.json           │
                                                                   │                    │
                                                    chunk cache (5)┤                    │
                                                                   v                    v
                                                         triplet mining (6) <───────────┘
                                                                   │
                                                          dataset_1.json
                                                                   │
                                                      fine-tuning (7, two stages)
                                                                   │
                                                          evaluation (8)
```

```bash
for court in atco1 cajp jcon jdgpj jsta jstj jtca jtcampca jtcampct jtcn jtrc jtre jtrg jtrl jtrp; do
    python -m crawler.dgsi.internal "$court"
done
python -m crawler.dgsi.external.stj
python -m crawler.dgsi.external.csm
python -m crawler.dgsi.external.pgr

python -m question_generation.create_request      # repeat for more decisions
python -m question_generation.request_status <batch_id>
python -m question_generation.retrieve_batch <output_file_id>

python -m processing.aggregator
python -m processing.standardize
python -m processing.prepare

python -m encoder.train.database add
python -m encoder.train.database standardize
mongoexport --uri "$MONGO_URI" --db database --collection encoder --jsonArray --out encoder/dataset.json
mongoexport --uri "$MONGO_URI" --db database --collection pool --jsonArray --out encoder/pool.json

python -m encoder.train.cache_processor
python -m helpers.triplet_generator
torchrun --nproc_per_node=<num_gpus> -m encoder.train.fine_tune --model encoder/models/albertina-1b5 --epochs 3 --lr 5e-6
# move encoder/results/finetuned-albertina-1b5-100k-triplets to encoder/models/
torchrun --nproc_per_node=<num_gpus> -m encoder.train.fine_tune

python -m encoder.test.clear
python -m encoder.test.test --model encoder/results/finetuned-albertina-1b5-100k-triplets
```

The step 4 legacy scripts (`helpers.find_distant`, `helpers.delete_wrong`, `helpers.embed`) are
left out above because the final model does not depend on them.

## Reproducibility

- **Seeds.** Triplet mining seeds Python, NumPy and PyTorch with 42. Fine-tuning seeds them with
  `--seed` (42) and also uses it for the dataset shuffle and the distributed sampler. Augmentation
  and random negatives inside DataLoader workers are seeded by PyTorch from the global seed.
- **Not bit-exact.** Training enables `cudnn.benchmark`, TF32 and reduced-precision reductions,
  and uses bf16 autocast. Mining runs the embedding model in fp16. Results will vary slightly
  between runs and hardware.
- **Snapshots.** The crawl, the `$sample` selections in question generation, and the LLM output
  cannot be reproduced exactly. Reproducing the thesis numbers needs the original exports
  (`dataset.json`, `pool.json`, `test.json`), `dataset_1.json` and the trained weights.
- **Recorded settings.** Training now writes `training_args.json` next to the model. Earlier
  thesis runs did not, so their settings come from the git history described above.
- **Environment.** `requirements.txt` pins the direct dependencies to the versions of the thesis
  environment.

## Results

No result files are part of the repository, and this README does not report the thesis numbers.
`encoder/test/test.py` writes `finetuned_evaluation_results.json` when run.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

The tests run on CPU without data, MongoDB or model downloads. They cover:

- the triplet loss
- the train/test split and the identical shuffle across ranks
- document chunking and the thread-local tokenizer
- the HTML parsers
- the embedding and id alignment in `helpers/embed.py`
- the evaluation subset selection
- the recall and rank computation

## Known limitations

- **Mined negatives are not used in training.** In `DatasetProcessor.__getitem__` the
  random-negative branch runs when `rand < 0.90 + 0.10`, which is always true. The last branch,
  which returns the triplet's own mined negative, can never be reached. So every negative the model
  sees is a random text from the train split, and the margin-controlled negatives from step 6 only
  enter that random pool. The data margin and `base_margin` are measured on these random
  negatives too. The thesis model was trained with this behaviour, so it was not changed. To use
  the mined negatives, the random branch would need `rand < random_neg_ratio` or a similar
  condition. That would be a change of method and would need new runs.
- **Possible overlap between training and evaluation documents.** Nothing in the code keeps the
  evaluation decisions (`test.json`) out of `dataset.json` or `pool.json`. If they overlap, the
  evaluation is optimistic.
- **False negatives.** Negatives are drawn from the pool without excluding chunks of the
  question's own decision. The pool is built from the same crawl, so a negative can come from the
  correct document or a near duplicate.
- **Inflated positive similarity.** When the best positive chunk scores below 0.25, the stored
  `pos_sim` is raised to 0.35, so `pos_sim` and `margin` in the triplet file are not always measured
  values.
- **E5 prefix mismatch.** Pool chunks are encoded without the `passage:` prefix, while positives
  have it. This shifts the similarities used to pick negatives.
- **Mixed embedding models for `wrong`.** `embed.py` and `find_distant.py` use different models
  and both write the same `embedding` field. This only affects the legacy `wrong` field.
- **Unused test split.** The 10% test split of the triplet file is never evaluated, and model
  selection uses training loss only.
- **Unguarded NaN skip.** If one GPU sees a NaN loss and skips its backward pass while the others
  do not, DDP can stall. The guard was kept as written.
- **Coverage.** Questions were generated for STJ decisions only, and `database add` also copies
  `files.jsta` documents that have questions. No script in the repository generates questions for
  `jsta`.
- **Single-pass chunking.** Chunks are built from tokenizer tokens and converted back to text, so
  whitespace and some characters can differ from the original decision.
- **Not verified after cleanup.** The cleanup was checked with unit tests, imports and `--help` of
  every entry point. No pipeline was run end to end because the database, data exports, model
  weights and GPUs were not available.

## Changes made during the archival cleanup

Bug fixes that can change outputs compared with the original code:

- **Evaluation.**
  - The document embedding cache used to be keyed by document id only. Evaluating a second model
    with the same cache directory reused the first model's document embeddings. The cache is now
    split per model and chunking setting.
  - The rank used for MRR now comes from the same fp16 scores as the top k.
- **Training.**
  - Each DDP process used to shuffle the train split with its own unseeded order, so the
    distributed sampler gave the processes overlapping and incomplete shards. The shuffle is now
    seeded.
  - The log reported a learning rate of 5e-6 while 2e-6 was used.
- **Chunk cache.** Worker threads shared and swapped one tokenizer. Failures from that race were
  counted as errors and the documents silently left out of the cache.
- **Legacy `wrong` field.** `helpers/embed.py` could pair an embedding with the wrong document id
  when some documents had no text, so `wrong` could point at the wrong document. It also reloaded
  the model for every document.
- **Crawlers.**
  - After a failed download the external crawlers restarted the whole crawl. They now skip the
    document.
  - The STJ crawler could copy the previous field's value into a field with no value.
  - The STJ crawler never stopped after the last page.
- **Triplet mining.** When the pool file was missing, the miner silently filled the pool with
  random vectors and "dummy_text" negatives. It now fails.

Other changes:

- The 15 DGSI crawler copies were merged into one module.
- `openai/` was renamed to `question_generation/` because the folder name could shadow the
  `openai` package.
- Hard-coded ids and paths became arguments.
- Removed scripts are kept in the git history:
  - `processing/clean.py` (its effect is included in `standardize.py`)
  - `openai/single_request.py` (a prototype that could not run)
  - `encoder/test/evaluation/test.py` (it only printed a record count)
  - `openai/requests.jsonl` (one batch input file)
- No-op flags `--neg-candidates` and `--similarity-batch` were removed from the miner.
- `requirements.txt` was a UTF-16 full freeze that lacked `ijson`. It now lists the direct
  dependencies.

## Citation

There is no published article yet. Please cite the thesis. The same information is in
`CITATION.cff`, which GitHub shows as "Cite this repository".

```bibtex
@mastersthesis{pereira2026foundational,
  author = {Pedro Pereira},
  title  = {Foundational Models for the Portuguese Legal Domain},
  school = {ESTG},
  year   = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). The license covers the code in this
repository only. The crawled decisions belong to the publishing courts and websites, the generated
questions are subject to the OpenAI terms, and fine-tuned weights derived from Albertina are
subject to Albertina's license. Check those terms before redistributing any data or weights.
