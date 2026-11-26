"""What every pipeline block is made of.

``INSIDE`` maps each block of the visualizer to the real functions and
classes that implement it, in the order a request goes through them.
``describe_inside`` resolves them with ``inspect`` so the page shows the
actual file, line, docstring and source code, never a copy that could
drift from the project.

Only the references listed here are ever resolved; the page cannot ask
for an arbitrary module.
"""

import importlib
import inspect
import os
from typing import Any, Dict, List, Tuple

Part = Tuple[str, str]  # (dotted reference, role in this block)
MAX_SOURCE_LINES = 140

INSIDE: Dict[str, Dict[str, List[Part]]] = {
    "corpus": {"all": [
        ("src.repositories.corpus_repository.CorpusRepository.discover",
         "walks data/raw/, keeps .py .md .rst .txt, skips .git etc."),
        ("src.repositories.corpus_repository.CorpusRepository."
         "read_with_digest",
         "reads one file and hashes it (incremental indexing)"),
    ]},
    "chunker": {
        "native": [
            ("src.services.chunkers.FileChunker.chunk",
             "dispatches on the file extension"),
            ("src.services.chunkers.PythonChunker.chunk",
             "ast.parse, then one chunk per top-level def / class"),
            ("src.services.chunkers.PythonChunker._spans",
             "walks the AST; big classes are split per method"),
            ("src.services.chunkers.MarkdownChunker.chunk",
             "one chunk per heading section"),
            ("src.services.chunkers.TextChunker.split_paragraphs",
             "cuts long sections on blank lines"),
            ("src.services.chunkers.TextChunker.chunk",
             "plain text and unparsable code"),
            ("src.services.span_splitter.SpanSplitter.merge_small",
             "merges small neighbours up to size * merge_ratio"),
            ("src.models.chunk.Chunk",
             "pydantic model: path, span, kind, context"),
        ],
        "langchain": [
            ("src.extra.langchain.chunker.LangChainChunker.language_of",
             "picks python / markdown / text"),
            ("src.extra.langchain.chunker.LangChainChunker.make_splitter",
             "builds RecursiveCharacterTextSplitter.from_language"),
            ("langchain_text_splitters.RecursiveCharacterTextSplitter"
             "._split_text",
             "LangChain: recursive split, separator by separator"),
            ("src.extra.langchain.chunker.LangChainChunker.split_spans",
             "start_index -> span; re-cuts pieces over the limit"),
            ("src.services.span_splitter.SpanSplitter.split_by_lines",
             "cuts a span on line boundaries under the limit"),
        ]},
    "embed": {"all": [
        ("src.bonus.semantic.embedder.Embedder.encode",
         "MiniLM: mean of token vectors, L2-normalized, 384-d"),
        ("src.bonus.semantic.semantic_indexer.SemanticIndexer.embedding_text",
         "chunk side: context + body (same encoder)"),
    ]},
    "index": {
        "native": [
            ("src.services.indexer.NativeIndexer.build",
             "chunks every file, tokenizes, builds and saves BM25"),
            ("src.services.indexer.NativeIndexer.searchable_text",
             "adds path words + context to the chunk text"),
            ("src.models.bm25_index.BM25Index.count_matrix",
             "sparse (chunks x terms) term-count matrix"),
            ("src.models.bm25_index.BM25Index.set_counts",
             "precomputes every BM25 weight W[d, t]"),
            ("src.repositories.bm25_repository.BM25Repository.save",
             "writes bm25_weights.npz + bm25_vocab.json"),
            ("src.repositories.chunk_repository.ChunkRepository.save",
             "writes chunks.jsonl"),
            ("src.repositories.meta_repository.MetaRepository.write",
             "writes meta.json (max_chunk_size, build id)"),
        ],
        "qdrant": [
            ("src.extra.qdrant.indexer.QdrantIndexer.build",
             "native chunks -> MiniLM vectors -> one point each"),
            ("src.extra.qdrant.indexer.QdrantIndexer._chunk_corpus",
             "reuses the native AST / heading chunkers"),
            ("src.bonus.semantic.embedder.Embedder.encode",
             "batches of 64 chunks -> 384-d unit vectors"),
            ("qdrant_client.QdrantClient.create_collection",
             "Qdrant: collection with cosine distance"),
            ("qdrant_client.QdrantClient.upsert",
             "Qdrant: points with span payload, 512 per batch"),
            ("src.repositories.meta_repository.MetaRepository.write",
             "writes meta.json (model, dim, max_chunk_size)"),
        ],
        "langchain": [
            ("src.extra.langchain.indexer.LangChainIndexer.build",
             "Documents with span metadata -> pickled retriever"),
            ("langchain_community.retrievers.BM25Retriever.from_texts",
             "LangChain: builds the retriever around BM25Okapi"),
            ("rank_bm25.BM25Okapi._calc_idf",
             "rank_bm25: idf of every term (with epsilon floor)"),
            ("src.repositories.meta_repository.MetaRepository.write",
             "writes meta.json (max_chunk_size, build id)"),
        ]},
    "query": {"all": [
        ("src.services.validator.InputValidator.query",
         "rejects empty / non-string queries"),
        ("src.services.validator.InputValidator.k",
         "k must be a positive integer"),
        ("src.services.rag_service.RagService.search",
         "validates, then searches through the cache"),
        ("src.bonus.cache.cached_search.CachedSearch.search",
         "LRU + SQLite cache in front of the retriever"),
    ]},
    "tokenize": {"all": [
        ("src.services.tokenizer.Tokenizer.tokenize",
         "words -> lower-case, stemmed terms, minus stopwords"),
        ("src.services.tokenizer.Tokenizer.split_identifier",
         "snake_case / camelCase -> sub-words"),
        ("src.services.tokenizer.Tokenizer.stem",
         "tiny suffix stripper"),
    ]},
    "retrieve": {
        "native": [
            ("src.services.retriever.LexicalRetriever.search",
             "top-k spans for a query"),
            ("src.services.retriever.LexicalRetriever.ranking",
             "tokenize + score + keep score > 0"),
            ("src.models.bm25_index.BM25Index.score",
             "sums the precomputed weight columns of the query terms"),
            ("src.services.ranker.Ranker.top_indices",
             "argpartition: k best without a full sort"),
            ("src.models.chunk.Chunk.to_source",
             "chunk -> MinimalSource for the grader"),
        ],
        "qdrant": [
            ("src.extra.qdrant.retriever.QdrantRetriever.scored",
             "embed the question, query_points, map payloads"),
            ("qdrant_client.QdrantClient.query_points",
             "Qdrant: nearest points by cosine"),
            ("qdrant_client.local.distances.cosine_similarity",
             "Qdrant embedded mode: exact numpy search, no HNSW"),
        ],
        "langchain": [
            ("src.extra.langchain.retriever.LangChainRetriever.search",
             "sets k, calls invoke, maps metadata to MinimalSource"),
            ("langchain_community.retrievers.BM25Retriever"
             "._get_relevant_documents",
             "LangChain: preprocess, then get_top_n"),
            ("rank_bm25.BM25Okapi.get_scores",
             "rank_bm25: Python loop over every document"),
        ]},
    "augment": {"all": [
        ("src.services.generator.AnswerGenerator.build_context",
         "packs ranked snippets into 3000 tokens"),
        ("src.repositories.corpus_repository.CorpusRepository."
         "read_span",
         "reads the text of one source"),
    ]},
    "llm": {"all": [
        ("src.services.generator.AnswerGenerator.__init__",
         "loads Qwen3-0.6B on cuda / mps / cpu"),
        ("src.services.generator.AnswerGenerator.answer",
         "chat template, greedy generate, strip <think>"),
        ("src.services.rag_service.RagService._answer_text",
         "answer cache keyed on model, prompt, index, sources"),
    ]},
    "output": {"all": [
        ("src.schemas.source.MinimalSource", "file_path + character span"),
        ("src.schemas.search.MinimalAnswer", "one answered question"),
        ("src.schemas.search.StudentSearchResultsAndAnswer",
         "the file answer_dataset writes"),
        ("src.services.rag_service.RagService.answer_dataset",
         "answers a whole search-results file"),
        ("src.repositories.json_repository.JsonRepository.save",
         "writes the JSON output file"),
    ]},
}

# the Qdrant backend reuses the native chunkers
INSIDE["chunker"]["qdrant"] = INSIDE["chunker"]["native"]


def resolve(ref: str) -> Any:
    """Import the longest module prefix of ``ref`` and walk attributes."""
    names = ref.split(".")
    for cut in range(len(names) - 1, 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(names[:cut]))
        except ImportError:
            continue
        for attr in names[cut:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"cannot import {ref}")


def _origin(ref: str) -> str:
    """Who wrote this code: the project or a library."""
    if ref.startswith("src."):
        return "project"
    return ref.split(".")[0]


def _location(obj: Any) -> Tuple[str, int]:
    """Short file path and first line of ``obj``."""
    path = inspect.getsourcefile(obj) or "?"
    _, line = inspect.getsourcelines(obj)
    if "site-packages" + os.sep in path:
        path = path.split("site-packages" + os.sep, 1)[1]
    else:
        path = os.path.relpath(path)
    return path, line


def describe(ref: str, role: str) -> Dict[str, Any]:
    """File, line, docstring and source code of one part."""
    names = ref.split(".")
    name = names[-1]
    if len(names) > 1 and names[-2][:1].isupper():
        name = f"{names[-2]}.{name}"
    info: Dict[str, Any] = {"ref": ref, "name": name, "role": role,
                            "origin": _origin(ref)}
    try:
        obj = resolve(ref)
        target = obj.__func__ if hasattr(obj, "__func__") else obj
        source = inspect.getsource(target)
        lines = source.splitlines()
        if len(lines) > MAX_SOURCE_LINES:
            lines = lines[:MAX_SOURCE_LINES] + ["    ..."]
        path, line = _location(target)
        doc = inspect.getdoc(target) or ""
        info.update(file=path, line=line, lines=len(source.splitlines()),
                    doc=doc.split("\n\n")[0], source="\n".join(lines))
    except Exception as err:  # noqa: BLE001 - show what failed
        info.update(error=f"{type(err).__name__}: {err}")
    return info


def describe_inside() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Every block, every backend, every part, fully resolved."""
    return {stage: {backend: [describe(ref, role) for ref, role in parts]
                    for backend, parts in variants.items()}
            for stage, variants in INSIDE.items()}
