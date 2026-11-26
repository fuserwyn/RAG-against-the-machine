*This project has been created as part of the 42 curriculum by fuserwyn.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about
the [vLLM](https://github.com/vllm-project/vllm) 0.10.1 codebase.

The pipeline:

1. **Indexing**: every Python / Markdown / text file of `data/raw/` is split
   into chunks (at most 2000 characters) and stored in a BM25 index under
   `data/processed/`.
2. **Retrieval**: a question is tokenized the same way and the top-k chunks
   are returned as source locations (`file_path` + character range).
3. **Augmenting**: the retrieved snippets are packed into the model context
   within a token budget.
4. **Generation**: `Qwen/Qwen3-0.6B` writes a short answer grounded in those
   snippets.

Retrieval quality is measured with recall@k (a reference source is found when
a retrieved span is in the same file and overlaps it with IoU ≥ 0.05).

The project ships **three interchangeable retrieval backends**, selected
with `--backend` (CLI) or `BACKEND=` (Makefile):

| Backend | Chunking | Retrieval |
|---|---|---|
| `native` (default) | own AST-based Python chunker, own heading-based Markdown chunker | own BM25 on a sparse matrix (numpy / scipy) |
| `langchain` | `RecursiveCharacterTextSplitter.from_language` (Python / Markdown) | `BM25Retriever` (langchain-community, rank_bm25) |
| `qdrant` | the native chunkers | MiniLM vectors in a Qdrant collection (vector database, embedded mode), cosine |

All five bonuses are implemented (native backend): semantic embeddings,
hybrid retrieval, incremental indexing, caching and a local HTTP API.

## Instructions

Requirements: Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/). The
deep-learning stack and model weights take a few GB of disk space.

```bash
make install          # uv sync
```

On a 42 computer the home quota is too small for PyTorch and the model
weights. From the clone:

```bash
make school           # goinfre, vLLM download, models, index, search, evaluate
make purge            # after the defense: venv, corpus, weights. The clone stays
```

`make school` finds a writable goinfre (`/sgoinfre/goinfre/Perso/$USER`,
`/sgoinfre/students/$USER`, `/sgoinfre/$USER` or `/goinfre/$USER`). The clone
stays where it is. The uv cache, the Python install, the Hugging Face cache
and, when the clone is on the home disk, `.venv` and `data/` are symlinks
onto that goinfre. After that, `uv sync` and
`uv run python -m src ...` from the subject write outside home.
`./scripts/school_setup.sh --help` lists
`--sgo`, `--zips`, `--no-prefetch` and `--yes`.

Expected layout (data is not versioned):

```
data/raw/vllm-0.10.1/                      # the corpus (unzip vllm-0.10.1.zip)
data/datasets/AnsweredQuestions/           # ground-truth datasets
data/datasets/UnansweredQuestions/         # questions only
data/processed/<backend>/                  # built by `index`
data/output/...                            # search / answer outputs
```

Makefile rules:

| Rule | Action |
|---|---|
| `make install` | install dependencies with uv |
| `make school` / `purge` | 42 computer: put the heavy data on sgoinfre / remove it |
| `make run` | `index` + `search` (both public datasets) + `evaluate` |
| `make index` / `search` / `evaluate` / `answer` | single pipeline steps |
| `make langchain` / `qdrant` | build those indexes; the visualizer reads them |
| `make update` | incremental re-index of the changed files (bonus) |
| `make serve` | local HTTP API on `http://127.0.0.1:$(PORT)` (bonus) |
| `make local` / `local-stop` | pipeline visualizer in the background, `http://127.0.0.1:8080` |
| `make up` / `down` | same visualizer in Docker |
| `make compare` | `run` with both backends |
| `make run MODE=semantic` / `MODE=hybrid` | build embeddings too and search with them (bonus) |
| `make debug` | run the indexer under `pdb` |
| `make clean` | remove caches |
| `make lint` / `make lint-strict` | flake8 + mypy |
| `make test` / `make test-fast` | pytest (all tests / without corpus and model tests) |

Variables: `BACKEND=native|langchain`, `MODE=lexical|semantic|hybrid`,
`K=10`, `CHUNK=2000`, `PORT=8000`, `SCOPE=UnansweredQuestions`.

## Example usage

```bash
# 1. index the corpus
uv run python -m src index --max_chunk_size 2000

# 2. single query
uv run python -m src search "How do I load a LoRA adapter at runtime?" --k 5

# 3. whole dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

# 4. self-evaluation (the official score comes from the moulinette)
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

# 5. answers
uv run python -m src answer "What HTTP endpoint loads a LoRA adapter?" --k 5
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions

# same commands with the LangChain backend
uv run python -m src --backend langchain index
uv run python -m src --backend langchain search "lora adapter"

# vector database backend (Qdrant, embedded: no server to start)
uv run python -m src --backend qdrant index
uv run python -m src --backend qdrant search "reduce GPU memory" --k 5

# bonuses
uv run python -m src index --semantic                     # BM25 + embeddings
uv run python -m src --mode hybrid search "reduce GPU memory" --k 5
uv run python -m src index --incremental                  # changed files only
uv run python -m src --cache False search "lora"          # bypass the cache
uv run python -m src serve --port 8000                    # HTTP API
curl "http://127.0.0.1:8000/search?query=lora%20adapter&k=3"
```

Global options (any command): `--backend native|langchain`,
`--mode lexical|semantic|hybrid`, `--cache True|False` and
`--index_dir data/processed` (each backend uses `<index_dir>/<backend>`).
Every path is a CLI argument; nothing is hard-coded.

Degenerate inputs (empty query, `k=0`, non-integer `k`, missing file,
malformed JSON, JSON with the wrong schema, missing or corrupted index,
unknown backend / mode, chunk size above 2000, unwritable output directory)
are reported as an `Error: ...` message with exit code 1, never as a
traceback. A nonsensical query is valid input: it returns no source.

## Pipeline visualizer

`visualizer/` holds a demo tool for the defense: an isometric, block-by-block
view of the pipeline that runs the real `src` code and shows every stage
(tokens, BM25 term contributions, token budget, prompt, Qwen3 answer) for
the `native`, `langchain` or `qdrant` backend, with a guided tour, the
index files and vocabulary, and a recorded MCP session (the engine starts
the MCP server and replays every JSON-RPC line it exchanged). It runs as two Docker containers
(nginx + Python engine) on a private `rag-net` network:

```bash
make up      # Docker, then open http://127.0.0.1:8080
make local       # same thing without Docker; the shell stays free
make local-stop  # stop that server
```

It is not part of the graded pipeline; see `visualizer/README.md`.

## System architecture

The code is split in three parts, so a reviewer sees at once what is
required, what is a bonus and what is extra:

- **mandatory core** (`models/`, `repositories/`, `services/`, `views/`):
  chunking, BM25, retrieval, Qwen3 answers, evaluation, CLI;
- **`bonus/`**: the five bonuses of the subject, one package each;
- **`extra/`**: two more backends (LangChain, Qdrant) that the subject does
  not ask for, kept to compare against the native one.

The core is layered: `schemas` (pydantic models of the subject) →
`models` (domain objects, no I/O) → `repositories` (all file I/O) →
`services` (logic) → `views` (CLI). Inside `bonus/` and `extra/` the file
names say the layer (`*_repository.py`, `*_indexer.py`, `*_retriever.py`).
Every module ends by creating the ready-to-use object
(`tokenizer = Tokenizer()`, `native_indexer = NativeIndexer(...)`, ...)
that the rest of the code imports; dependencies are passed through
constructors, so tests can swap any of them.

```
src/
├── __main__.py              # python -m src -> views.cli.main
├── config.py                # Settings: default paths, limits, models
├── errors.py                # RagError: user-facing errors
├── schemas/                 # pydantic models of the subject
│   ├── source.py            # MinimalSource
│   ├── question.py          # UnansweredQuestion, RagDataset, ...
│   └── search.py            # MinimalSearchResults, MinimalAnswer, ...
├── models/                  # domain objects, no I/O
│   ├── chunk.py             # Chunk (span + kind + context)
│   ├── bm25_index.py        # BM25Index: the sparse BM25 math (matrix W)
│   ├── index_build.py       # BuildSettings, IndexReport
│   └── evaluation_report.py # recall@k report
├── repositories/            # all file I/O
│   ├── corpus_repository.py # data/raw: discovery, reading, spans
│   ├── chunk_repository.py  # chunks.jsonl
│   ├── bm25_repository.py   # bm25_weights.npz, bm25_counts.npz, vocab
│   ├── meta_repository.py   # meta.json, build id
│   └── json_repository.py   # dataset / result JSON files
├── services/                # the mandatory pipeline
│   ├── tokenizer.py         # identifier-aware tokenizer
│   ├── span_splitter.py     # line / merge span arithmetic
│   ├── chunkers.py          # Python (AST), Markdown, text chunkers
│   ├── indexer.py           # corpus -> chunks.jsonl + BM25 matrix W
│   ├── ranker.py            # top-k of a score vector
│   ├── search_backend.py    # SearchBackend protocol: search(query, k)
│   ├── retriever.py         # LexicalRetriever: BM25 top-k
│   ├── generator.py         # Qwen3-0.6B answer generation
│   ├── evaluator.py         # recall@k with IoU >= 0.05
│   ├── validator.py         # argument validation
│   ├── retriever_factory.py # backend + mode -> retriever
│   └── rag_service.py       # RagService: the use cases the CLI calls
├── views/
│   └── cli.py               # Fire CLI, error handling
├── bonus/
│   ├── semantic/            # 1. MiniLM embeddings
│   │   ├── embedder.py, embedding_repository.py
│   │   └── semantic_indexer.py, semantic_retriever.py
│   ├── hybrid/              # 2. BM25 + semantic with RRF
│   │   └── rank_fusion.py, hybrid_retriever.py
│   ├── incremental/         # 3. re-index only changed files
│   │   ├── manifest_repository.py       # sha1 of every file
│   │   └── previous_build.py, previous_build_loader.py
│   ├── cache/               # 4. SQLite + LRU cache
│   │   └── cache_repository.py, cached_search.py
│   └── http_api/            # 5. local HTTP API
│       └── api.py
└── extra/                   # not required by the subject
    ├── langchain/           # RecursiveCharacterTextSplitter + BM25Retriever
    │   └── langchain_repository.py, chunker.py, indexer.py, retriever.py
    ├── qdrant/              # MiniLM vectors in embedded Qdrant
    │   └── qdrant_repository.py, indexer.py, retriever.py
    └── mcp_server/          # the BM25 search as MCP tools (agentic RAG)
        └── tools.py, server.py
tests/                       # pytest, one module per subject section
```

Two places where the core calls a bonus: `services/indexer.py` asks
`bonus/incremental` for the previous build when `index --incremental` is
used (the reuse itself changes how the matrix is assembled, so it stays in
the indexer), and `rag_service.py` wraps the retriever in `bonus/cache`
and builds the semantic index on `index --semantic`.

Data flow: `index` → `data/processed/<backend>/` → `search` /
`search_dataset` (→ `StudentSearchResults` JSON) → `answer_dataset`
(→ `StudentSearchResultsAndAnswer` JSON). All exchanged structures are
validated by pydantic. Every retriever exposes the same
`search(query, k) -> List[MinimalSource]` method; `RagService` wraps the
selected one in a cache, and both the CLI and the HTTP API only call
`RagService`. The LangChain modules and torch are only imported when they
are needed.

## Chunking strategy

Every chunk is an exact `[first_character_index, last_character_index)` span
of the original file (read without newline translation, so `\r\n` files keep
exact offsets) and is never longer than `--max_chunk_size`.

**Python (native).** The file is parsed with `ast`. It is cut at the start of
every top-level statement: the start includes its decorators and the comment
lines directly above it, so each function or class keeps its leading
comments and decorators. A class or function that is still too long is cut
again at its own body statements (methods), and anything left over is split
on line boundaries. Small neighbouring pieces with the same enclosing class
are merged up to half of the maximum size. The enclosing class / function
name is stored as context and indexed. Files that do not parse fall back to
the text chunker.

**Markdown (native).** The file is cut at every heading (headings inside
fenced code blocks are ignored). The heading path (`LoRA > Serving`) is kept
as context. Sections over the limit are split on blank lines, then on lines;
small consecutive sections are merged.

**LangChain.** `RecursiveCharacterTextSplitter.from_language` with the
`PYTHON` separators (`class`, `def`, blank lines, …) or the `MARKDOWN`
separators (headings, code fences, …), `chunk_overlap=0`. The splitter only
reports `start_index`; the end is `start + len(chunk)`, and any piece still
above the limit is re-cut on lines.

## Retrieval method

Both backends use **Okapi BM25** (k1 = 1.2, b = 0.75) over the same tokenizer:

- words are extracted with a regex, lower-cased, stop words removed,
  light plural stemming applied (`formats` → `format`);
- identifiers are kept whole **and** split into parts:
  `PagedAttentionBackend` → `pagedattentionbackend, paged, attention,
  backend`; `max_num_seqs` → `max_num_seq, max, num, seq`. A question can
  therefore match either a verbatim identifier or a paraphrase;
- the file path words (e.g. `docs features lora md`) and the chunk context
  (class name / heading path) are indexed together with the chunk text.

In the native backend all BM25 weights
`idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))` are precomputed into a
sparse CSC matrix (chunks × terms). Scoring a query is summing a few columns,
then `argpartition` gives the top k. The LangChain backend uses
`BM25Retriever` with the same tokenizer as `preprocess_func`.

## Bonus 1: semantic embeddings

Next to the BM25 index, the native backend can build a dense vector index
(`src/bonus/semantic/`):

- every chunk (context + body, truncated to 256 tokens) is encoded by
  `sentence-transformers/all-MiniLM-L6-v2` with mean pooling, then
  L2-normalized → a `23 352 × 384` float32 matrix saved as
  `data/processed/native/embeddings.npy` (~36 MB);
- the question is encoded the same way; since all vectors have length 1,
  cosine similarity with every chunk is one matrix-vector product
  (`embeddings @ q`), then `argpartition` gives the top k;
- no external vector database: at this size a numpy matrix is enough;
- the embeddings record the `build_id` of the lexical index they belong to:
  stale embeddings are refused (and a plain re-index deletes them).

It is opt-in (`index --semantic`) because embedding the corpus takes
~100 s on Apple MPS and longer on a CPU-only machine, while plain indexing
takes ~7 s.

Alone, MiniLM is clearly weaker than BM25 here (see the table below):
questions quote exact identifiers (`max_num_seqs`, `/v1/load_lora_adapter`)
that BM25 matches verbatim, MiniLM was not trained on code, and it only
reads the first 256 tokens of a chunk. Its strength is paraphrases with no
shared words, which is what hybrid retrieval uses.

## Bonus 2: hybrid retrieval

`--mode hybrid` (`src/bonus/hybrid/`) merges the BM25 and the MiniLM
rankings (top 100 of each) with weighted **Reciprocal Rank Fusion**:

```
fused(d) = 1.0 / (30 + rank_bm25(d)) + 0.2 / (30 + rank_minilm(d))
```

Ranks are used instead of raw scores because the two scales are not
comparable (BM25 is unbounded, cosine is in [-1, 1]). The weights come from
a small grid search on the public datasets (recall@5, code / docs):

| rrf_k | semantic weight | code | docs |
|---|---|---|---|
| – | 0 (BM25 only) | 0.848 | 0.860 |
| 60 | 1.0 (classic RRF) | 0.758 | 0.780 |
| 60 | 0.5 | 0.798 | 0.800 |
| 30 | 0.5 | 0.828 | 0.830 |
| **30** | **0.2** | **0.869** | **0.870** |
| 10 | 0.3 | 0.869 | 0.870 |

With equal weights the weak semantic ranking pulls good BM25 results down;
with a small weight it mainly breaks ties and promotes paraphrase matches
that BM25 ranked 6th–10th. A plateau of settings reaches the same score, so
the chosen one is not a lucky single point, but it was tuned on the public
sets and may be slightly optimistic on the private ones.

## Bonus 3: incremental indexing

`index --incremental` (`make update`) re-indexes only what changed:

- every build stores a manifest (`manifest.json`) with the SHA-1 of each
  file and the raw BM25 term counts (`bm25_counts.npz`, chunks × terms);
- an incremental build hashes the corpus files, reuses the chunks and the
  count rows of unchanged files, re-chunks and re-tokenizes only the added
  or modified files, drops removed files, and recomputes the BM25 weights
  from the merged counts (IDF and average length depend on the whole
  corpus, so this step is global but it is pure vectorized math);
- the result is identical to a full rebuild (checked by the tests);
- if embeddings exist, only the chunks of changed files are embedded again,
  the other vectors are copied;
- a different `--max_chunk_size` / corpus directory, or a missing or
  damaged previous index, falls back to a full build.

On the vLLM corpus: full lexical build ~7 s, incremental with no change
~1.3 s; with embeddings, one modified file re-embeds 12 chunks in ~30 s
(mostly loading MiniLM) instead of ~108 s for all 23 352 chunks.

## Bonus 4: caching

- **Index**: everything expensive is computed once at indexing time and
  persisted (BM25 weights as a sparse matrix, embeddings as a numpy
  matrix), so loading an index takes ~0.1 s.
- **Query results**: `src/bonus/cache/cache_repository.py` stores search results and
  generated answers in a small SQLite file next to the index
  (`query_cache.sqlite`) with an in-memory LRU in front. Keys contain the
  backend, mode, `k`, the query and the index `build_id`; answers also use
  the model name, the system prompt and the exact source spans. Rebuilding
  the index clears the cache, and old keys could not match anyway.
- **Cold start**: the retriever is loaded lazily, only on a cache miss, so
  a repeated query does not load the index, MiniLM or Qwen at all.

Measured on the development laptop (process start included):

| Command | miss | hit |
|---|---|---|
| `--mode hybrid search ...` | 6–11 s (loads MiniLM) | 0.2–0.4 s |
| `answer ...` | ~19 s (loads Qwen) | ~0.5 s |

`--cache False` disables it (Fire does not accept `--nocache` before the
command name). Any cache failure (read-only disk, corrupted file) silently
falls back to computing the result.

## Bonus 5: local HTTP API

`serve` (`make serve`) starts a small threaded HTTP server built on the
standard library (`src/bonus/http_api/api.py`), bound to `127.0.0.1:8000` by default:

| Endpoint | Input | Output |
|---|---|---|
| `GET /health` | – | `{"status": "ok", "backend", "mode", "index_dir"}` |
| `GET /search?query=...&k=5`, `POST /search` | `{"query": "...", "k": 5}` | `MinimalSearchResults` |
| `GET /answer?query=...&k=5`, `POST /answer` | `{"query": "...", "k": 5, "model_name": "..."}` | `MinimalAnswer` |

```bash
uv run python -m src --mode hybrid serve --port 8000
curl -X POST http://127.0.0.1:8000/answer \
     -H 'Content-Type: application/json' \
     -d '{"query": "How do I load a LoRA adapter at runtime?", "k": 3}'
```

The index (and the model, on the first `/answer`) is loaded once and shared
by all requests; generation is serialized with a lock. Invalid requests get
`400 {"error": "..."}` with the same validation as the CLI, unknown paths
`404`, wrong methods `405`; an unexpected error is a `500` and the server
keeps running. Ctrl+C stops it cleanly.

## Performance analysis

Public datasets (99 code questions, 100 docs questions), measured with the
`evaluate` command on an Apple Silicon laptop.

Default configuration (`--max_chunk_size 2000`):

| Backend / mode | Dataset | R@1 | R@3 | **R@5** | R@10 | Required R@5 |
|---|---|---|---|---|---|---|
| native lexical | code | 0.566 | 0.788 | **0.848** | 0.919 | 0.50 |
| native lexical | docs | 0.680 | 0.840 | **0.860** | 0.890 | 0.80 |
| native semantic | code | 0.263 | 0.444 | **0.515** | 0.626 | – |
| native semantic | docs | 0.340 | 0.520 | **0.620** | 0.680 | – |
| native hybrid | code | 0.556 | 0.778 | **0.869** | 0.909 | – |
| native hybrid | docs | 0.620 | 0.790 | **0.870** | 0.910 | – |
| langchain | code | 0.525 | 0.747 | **0.788** | 0.879 | 0.50 |
| langchain | docs | 0.660 | 0.820 | **0.870** | 0.890 | 0.80 |
| qdrant | code | 0.263 | 0.444 | **0.515** | 0.626 | – |
| qdrant | docs | 0.340 | 0.520 | **0.620** | 0.680 | – |

`qdrant` scores exactly like `native semantic`: same chunks, same MiniLM
vectors, same exact cosine search. Only the storage changes (a vector
database instead of a numpy file). Pure vector search misses the docs
threshold, so the graded configuration stays BM25 (`native`).

Effect of the chunk size on recall@5 (code / docs):

| Chunk size | native chunks | native R@5 | langchain chunks | langchain R@5 |
|---|---|---|---|---|
| 500 | 80 949 | 0.869 / 0.880 | 58 944 | 0.828 / 0.860 |
| 1000 | 44 152 | 0.889 / 0.840 | 29 296 | 0.798 / 0.860 |
| 2000 | 23 352 | 0.848 / 0.860 | 14 496 | 0.788 / 0.870 |

Smaller chunks are more precise for code questions (a question is usually
about one function), while docs questions are less sensitive. Larger chunks
give the generator more context per source, so 2000 stays the default.

Speed:

- indexing the whole corpus (1 968 files): ~7 s (native), ~10 s (langchain),
  +~100 s for the embeddings;
- searching 100 questions: < 0.5 s (native lexical, process start
  included), ~3–4 s (langchain: `rank_bm25` scores documents in a Python
  loop), ~7–12 s (hybrid, mostly loading MiniLM);
- answer generation: a single `answer` call takes ~19 s on the development
  laptop, most of it loading Qwen3-0.6B; `answer_dataset` loads the model
  once for the whole dataset.

## Testing

```bash
make test        # every test, 315 of them (~2 min)
make test-fast   # without the corpus / model tests (< 1 min)
```

`tests/` has one module per section of the subject, e.g.
`test_v_common_instructions.py` (flake8, mypy, docstrings, type hints,
context managers, Makefile rules), `test_vi1_indexing.py` (both chunkers,
BM25 formula, size limits, 5-minute indexing), `test_vi2_retrieval.py`
(exact paths, recall@5 thresholds, 200 questions under 90 s),
`test_vi6_cli.py` (every degenerate input), `test_vii_evaluation.py`
(IoU / recall@k), `test_viii_readme.py`, and one module per bonus. Fast tests
run on a small synthetic corpus with fake embedder / generator classes.
Tests marked `corpus` need `data/raw` and `data/datasets`; tests marked
`model` load the real Qwen3-0.6B / MiniLM weights. Both are skipped
automatically when the data or the weights are not available.

## Bonus-adjacent: a vector database backend (`--backend qdrant`)

The semantic bonus only asks for a vector index; `native --mode semantic`
keeps it as a numpy matrix. The `qdrant` backend shows the same idea with
a real vector database:

- `src/extra/qdrant/indexer.py` chunks with the native chunkers, embeds every
  chunk with all-MiniLM-L6-v2 and upserts one point per chunk; the payload
  holds `file_path` and the character span, so a hit maps straight to a
  `MinimalSource`.
- `src/extra/qdrant/retriever.py` embeds the question and calls
  `query_points` (cosine).
- Qdrant runs in **embedded mode** (`QdrantClient(path=...)`): no server or
  container, the collection lives in `data/processed/qdrant/storage`.
  Embedded mode searches exactly in numpy (no HNSW graph); at ~23k points
  that takes about 10 ms. It also locks its storage, so one process at a
  time can use the `qdrant` backend (a second one gets a clear error).

## Extra: an MCP server (agentic RAG)

The pipeline above is *fixed*: one question → one search → one prompt.
`src/extra/mcp_server/` exposes the same BM25 index to any
[MCP](https://modelcontextprotocol.io) client, so the **client's model**
decides when to search, with which words,
and how many times: it can search again with identifiers it found in a
first hit, then read the code around it.

| tool | arguments | returns |
|---|---|---|
| `search_vllm` | `query`, `k` (1..20, default 5) | the k best chunks: rank, file path, character span, text |
| `read_source` | `file_path`, `start`, `end` | at most 6000 characters of a corpus file |

- `tools.py` holds the logic (`RagTools`), independent of the protocol: it
  reuses `RagService`, reads the text of each hit (the index only stores
  addresses), validates `k` and keeps `read_source` inside `data/raw`
  (`..`, absolute paths and directories are refused).
- `server.py` registers both functions on an `MCPServer`; the SDK turns the
  type hints into JSON schemas (`tools/list`) and the pydantic return types
  into structured results. A `RagError` becomes a `ToolError`, so the model
  reads the reason instead of a crash.
- Transport is stdio (JSON-RPC on stdin/stdout); the index is loaded on the
  first call (~150 ms), then a search takes a few ms.

```bash
uv run python -m src index                 # the native index is needed
uv run python -m src.extra.mcp_server      # stdio: the client owns the process
```

Point the client at that command. A typical config entry:

```json
{
  "mcpServers": {
    "vllm-rag": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/this/repo",
               "python", "-m", "src.extra.mcp_server"]
    }
  }
}
```

`RAG_INDEX_DIR` / `RAG_RAW_DIR` override the index and corpus directories.
`tests/test_extra_mcp_server.py` starts the server as a subprocess and talks
to it with the MCP client of the SDK.

## Design decisions

- **BM25 over TF-IDF**: term-frequency saturation and length normalization
  prevent long chunks full of repeated tokens from dominating.
- **Identifier splitting** is the single most useful trick: questions are
  written in English while the answer lives in `snake_case` / `CamelCase`
  code.
- **Indexing path and context words** lets `"LoRA"` in a question hit
  `docs/features/lora.md` even when the chunk body never says "LoRA".
- **Syntax-aware chunk boundaries** keep a function with its docstring,
  decorators and leading comments, which is what a reference span usually
  covers.
- **Precomputed sparse weights** make retrieval a matrix column sum, far
  below the 90 s / 200 questions limit. Keeping the raw counts too is what
  makes incremental indexing cheap.
- **Two backends behind one interface** (`search(query, k)`), selected at
  runtime, so they can be compared on the same datasets.
- **Rank fusion with a small semantic weight**, because the semantic
  ranking alone is much weaker than BM25 on this corpus.
- **A build id per index** ties embeddings and cached results to the exact
  index they were computed from.
- **Standard library HTTP server** instead of a web framework: four
  endpoints do not justify a new dependency.
- **Generation prompt**: a system prompt that requires answering only from
  the numbered context, quoting exact identifiers, and saying so when the
  context lacks the answer; Qwen3 "thinking" is disabled and decoding is
  greedy with a small repetition penalty for stable, short answers. Context
  is capped at 3000 tokens. When nothing is retrieved the model is not
  even loaded.

## Challenges faced

- **Exact character offsets**: some files use `\r\n`; files are read without
  newline translation so offsets are not shifted.
- **Chunks over 2000 characters** make the whole output invalid, so every
  path (AST pieces, long lines, LangChain chunks) ends with a hard
  line-based split.
- **Vocabulary mismatch** between questions and code: solved with identifier
  splitting, light stemming and indexing path / context words.
- **Robust CLI**: Fire passes raw values (`--k abc` becomes a string), so
  every argument is validated and all errors end up as a clean message.
  Output directories are created before the work starts, so a bad path does
  not waste a long `answer_dataset` run.
- **Hybrid retrieval was first worse than BM25** (classic RRF: 0.758 /
  0.780 recall@5); weighting the rankings fixed it.
- **Stale derived data**: after re-indexing with another chunk size, old
  embeddings no longer matched the chunks; the build id makes this an
  explicit error instead of silently wrong results.
- **mypy and recent numpy stubs**: pinning `python_version = 3.10` in the
  mypy config made it fail on numpy's 3.12-only stub syntax; mypy now uses
  the interpreter's version.
- **Low disk space** during development: heavy dependencies (torch,
  transformers, LangChain) are imported lazily so the native retrieval works
  on its own.

## Resources

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks* (2020): https://arxiv.org/abs/2005.11401
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond* (2009)
- Manning, Raghavan, Schütze, *Introduction to Information Retrieval*,
  chapters 6 and 11: https://nlp.stanford.edu/IR-book/
- Cormack, Clarke, Büttcher, *Reciprocal Rank Fusion outperforms Condorcet
  and individual Rank Learning Methods* (SIGIR 2009)
- Reimers & Gurevych, *Sentence-BERT* (2019) and the all-MiniLM-L6-v2 model
  card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Python `ast` module: https://docs.python.org/3/library/ast.html
- Python `http.server` and `sqlite3` modules:
  https://docs.python.org/3/library/http.server.html
- LangChain text splitters:
  https://python.langchain.com/docs/concepts/text_splitters/
- LangChain BM25 retriever:
  https://python.langchain.com/docs/integrations/retrievers/bm25/
- Hugging Face Transformers, Qwen3 model card:
  https://huggingface.co/Qwen/Qwen3-0.6B
- Python Fire: https://github.com/google/python-fire
- Pydantic: https://docs.pydantic.dev/

### Use of AI

An AI coding assistant was used to:

- scaffold the project structure, Makefile and pyproject;
- draft the chunkers, BM25 implementation, CLI and the LangChain backend;
- draft the bonus modules (hybrid fusion, incremental indexing, cache,
  HTTP API) and the pytest suite;
- run the evaluation experiments (chunk size comparison, hybrid weight grid
  search) and fix lint issues;
- draft this README.

All generated code was read, run and checked against the public datasets;
design choices (identifier splitting, context indexing, chunk boundaries,
fusion weights) were reviewed so they can be explained and modified during
the defense.
