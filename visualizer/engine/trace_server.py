"""Trace server for the RAG visualizer.

Runs the real pipeline of the project mounted at the working directory
(``src/`` + ``data/processed``) one stage at a time and reports what every
stage did: tokens, BM25 scores with per-term contributions, the context
packed for the model, the chat prompt and the generated answer.

Endpoints (JSON):
    GET  /api/health
    GET  /api/stats
    GET  /api/chunks    ?backend=&offset=&limit=&q=  (stored chunk records)
    GET  /api/matrix    ?row=  (one row of the native BM25 matrix W)
    POST /api/trace    {"query": str, "k": int, "backend": str}
    POST /api/generate  {"query": str, "sources": [MinimalSource, ...]}
    POST /api/mcp       {"query": str, "k": int}  (a recorded MCP session)
"""

import argparse
import bisect
import collections
import json
import os
import pickle
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs

import numpy as np

from src.repositories.corpus_repository import corpus_repository
from src.repositories.meta_repository import meta_repository
from src.extra.qdrant.qdrant_repository import qdrant_repository
from src.schemas import MinimalAnswer, MinimalSource
from src.services.generator import (DEFAULT_MODEL, NO_CONTEXT_ANSWER,
                                    SYSTEM_PROMPT)
from src.services.indexer import NativeIndexer
from src.services.retriever import LexicalRetriever
from src.services.tokenizer import tokenizer

from inside import describe_inside
from mcp_trace import record_session

NATIVE_DIR = "data/processed/native"
LANGCHAIN_FILE = "data/processed/langchain/bm25_retriever.pkl"
QDRANT_DIR = "data/processed/qdrant"
COLLECTION = qdrant_repository.collection
MAX_WORDS = 400                 # words sent for the W animation
VOCAB_ROWS = 40                 # chunks listed for one term
FILE_HEAD = 12                  # first values shown per stored array
MAX_CONTEXT_TOKENS = 3000
PREVIEW_CHARS = 1500
BACKENDS = ("native", "langchain", "qdrant")


class BadRequest(Exception):
    """Invalid input from the client."""


def compact_json(obj: Any) -> str:
    """One-line JSON, the way chunks.jsonl stores its records."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def ms(start: float) -> float:
    """Milliseconds elapsed since ``start``."""
    return round((time.perf_counter() - start) * 1000.0, 2)


class Engine:
    """Lazily loaded indices, tokenizer and model shared by all requests."""

    def __init__(self) -> None:
        """Start empty; everything is loaded on first use."""
        self.lock = threading.Lock()
        self.gen_lock = threading.Lock()
        self.native: Optional[LexicalRetriever] = None
        self.lc: Any = None
        self.lc_pos: Dict[int, int] = {}
        self.qd: Any = None
        self.qwen_tok: Any = None
        self.generator: Any = None
        self.texts: Dict[str, str] = {}
        self.load_times: Dict[str, float] = {}
        self.inside: Optional[Dict[str, Any]] = None
        self.stores: Dict[str, Dict[str, Any]] = {}
        self.mx: Optional[Dict[str, Any]] = None
        self.disk: Optional[Dict[str, Any]] = None

    # ---- loading -------------------------------------------------------
    def get_native(self) -> LexicalRetriever:
        """The native BM25 retriever."""
        with self.lock:
            if self.native is None:
                start = time.perf_counter()
                self.native = LexicalRetriever(NATIVE_DIR)
                self.load_times["native"] = ms(start)
            return self.native

    def get_langchain(self) -> Any:
        """The pickled LangChain BM25Retriever."""
        with self.lock:
            if self.lc is None:
                start = time.perf_counter()
                with open(LANGCHAIN_FILE, "rb") as handle:
                    self.lc = pickle.load(handle)
                self.lc_pos = {id(doc): i
                               for i, doc in enumerate(self.lc.docs)}
                self.load_times["langchain"] = ms(start)
            return self.lc

    def get_qdrant(self) -> Any:
        """The Qdrant retriever (opens the embedded storage + MiniLM).

        Loaded only on demand: embedded Qdrant locks its storage, so
        while it is open the CLI cannot use ``--backend qdrant``.
        """
        with self.lock:
            if self.qd is None:
                from src.extra.qdrant.retriever import QdrantRetriever
                start = time.perf_counter()
                self.qd = QdrantRetriever(QDRANT_DIR)
                self.load_times["qdrant"] = ms(start)
            return self.qd

    def get_tokenizer(self) -> Any:
        """The Qwen3 tokenizer (small; no model weights)."""
        with self.lock:
            if self.qwen_tok is None:
                from transformers import AutoTokenizer
                self.qwen_tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
            return self.qwen_tok

    def get_generator(self) -> Any:
        """The project's AnswerGenerator (loads Qwen3-0.6B weights)."""
        with self.lock:
            if self.generator is None:
                from src.services.generator import AnswerGenerator
                start = time.perf_counter()
                self.generator = AnswerGenerator(DEFAULT_MODEL)
                self.load_times["qwen3"] = ms(start)
            return self.generator

    def text_of(self, path: str) -> str:
        """File content, cached."""
        if path not in self.texts:
            if len(self.texts) > 256:
                self.texts.clear()
            self.texts[path] = corpus_repository.read_text(path)
        return self.texts[path]

    # ---- stats ---------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Index statistics for both backends."""
        out: Dict[str, Any] = {}
        try:
            nat = self.get_native()
            kinds = collections.Counter(c.kind for c in nat.chunks)
            sizes = [c.last_character_index - c.first_character_index
                     for c in nat.chunks]
            out["native"] = {
                "chunks": len(nat.chunks),
                "files": len({c.file_path for c in nat.chunks}),
                "vocab": len(nat.bm25.vocab),
                "kinds": dict(kinds.most_common()),
                "avg_chars": round(float(np.mean(sizes)), 1),
                "max_chars": int(max(sizes)),
                "max_chunk_size": meta_repository.read(
                    NATIVE_DIR).get("max_chunk_size"),
                "k1": nat.bm25.k1, "b": nat.bm25.b,
                "load_ms": self.load_times.get("native"),
            }
        except Exception as err:  # noqa: BLE001 - report, do not crash
            out["native"] = {"error": f"{type(err).__name__}: {err}"}
        try:
            lc = self.get_langchain()
            vec = lc.vectorizer
            sizes = [d.metadata["last_character_index"]
                     - d.metadata["first_character_index"] for d in lc.docs]
            out["langchain"] = {
                "chunks": len(lc.docs),
                "files": len({d.metadata["file_path"] for d in lc.docs}),
                "vocab": len(vec.idf),
                "avg_chars": round(float(np.mean(sizes)), 1),
                "max_chars": int(max(sizes)),
                "max_chunk_size": meta_repository.read(
                    os.path.dirname(LANGCHAIN_FILE)).get("max_chunk_size"),
                "k1": vec.k1, "b": vec.b,
                "vectorizer": type(vec).__name__,
                "retriever": type(lc).__name__,
                "load_ms": self.load_times.get("langchain"),
            }
        except Exception as err:  # noqa: BLE001
            out["langchain"] = {"error": f"{type(err).__name__}: {err}"}
        meta = meta_repository.read(QDRANT_DIR)
        if meta:
            out["qdrant"] = {
                "chunks": meta.get("n_chunks"), "dim": meta.get("dim"),
                "model": meta.get("model"), "distance": meta.get("distance"),
                "max_chunk_size": meta.get("max_chunk_size"),
                "storage": "embedded (local mode)",
                "load_ms": self.load_times.get("qdrant"),
            }
        else:
            out["qdrant"] = {"error": "no Qdrant index: run "
                             "'--backend qdrant index'"}
        return out

    # ---- stored chunks -------------------------------------------------
    def chunk_store(self, backend: str) -> Dict[str, Any]:
        """Every chunk record of a backend, as the JSON it is stored as."""
        with self.lock:
            cached = self.stores.get(backend)
        if cached is not None:
            return cached
        if backend == "native":
            path = os.path.join(NATIVE_DIR, "chunks.jsonl")
            with open(path, encoding="utf-8") as handle:
                lines = [line.rstrip("\n") for line in handle if line.strip()]
            store = {"store": path, "bytes": os.path.getsize(path),
                     "format": "JSON Lines: one chunk per line. Line number "
                               "= chunk id = row of the BM25 matrix and of "
                               "embeddings.npy. No text, only the span.",
                     "lines": lines}
        elif backend == "langchain":
            lc = self.get_langchain()
            lines = []
            for doc in lc.docs:
                text = doc.page_content
                short = text[:60] + ("…" if len(text) > 60 else "")
                lines.append(compact_json({"metadata": doc.metadata,
                                           "page_content": short}))
            store = {"store": LANGCHAIN_FILE + " → retriever.docs[i]",
                     "bytes": os.path.getsize(LANGCHAIN_FILE),
                     "format": "LangChain Document objects inside one "
                               "pickle, shown here as JSON. The full text "
                               "is stored too (page_content, cut here).",
                     "lines": lines}
        else:
            qd = self.get_qdrant()
            points: List[Any] = []
            offset: Any = None
            while True:
                batch, offset = qd.client.scroll(
                    COLLECTION, limit=4096, offset=offset,
                    with_payload=True, with_vectors=False)
                points.extend(batch)
                if offset is None:
                    break
            points.sort(key=lambda p: int(p.id))
            vector = f"<{qd.meta.get('dim', '?')} float32, not shown>"
            lines = [compact_json({"id": p.id, "payload": p.payload,
                                   "vector": vector})
                     for p in points]
            store = {"store": f"{QDRANT_DIR}/storage → collection "
                              f"'{COLLECTION}'",
                     "bytes": None,
                     "format": "Qdrant points: id + payload (the span) + "
                               "the 384-d vector. The vector is what the "
                               "search compares; the payload is returned.",
                     "lines": lines}
        with self.lock:
            self.stores[backend] = store
        return store

    def _record_text(self, line: str) -> str:
        """The text a stored record points to, read from data/raw."""
        try:
            rec = json.loads(line)
            src = rec.get("payload") or rec.get("metadata") or rec
            path = str(src["file_path"])
            first = int(src["first_character_index"])
            last = int(src["last_character_index"])
            return self.text_of(path)[first:last]
        except (OSError, ValueError, KeyError, TypeError):
            return ""

    def chunks_page(self, backend: str, offset: int, limit: int,
                    query: str) -> Dict[str, Any]:
        """One page of stored records, optionally filtered by a substring."""
        store = self.chunk_store(backend)
        lines: List[str] = store["lines"]
        if query:
            needle = query.lower()
            ids = [i for i, line in enumerate(lines)
                   if needle in line.lower()]
        else:
            ids = list(range(len(lines)))
        items = [{"n": i, "line": lines[i],
                  "text": self._record_text(lines[i])}
                 for i in ids[offset:offset + limit]]
        return {"backend": backend, "store": store["store"],
                "bytes": store["bytes"], "format": store["format"],
                "total": len(lines), "matched": len(ids), "offset": offset,
                "limit": limit, "query": query, "items": items}

    # ---- BM25 matrix ----------------------------------------------------
    def matrix_parts(self) -> Dict[str, Any]:
        """W as CSR (fast rows), the raw counts, df, idf and doc lengths."""
        nat = self.get_native()
        with self.lock:
            if self.mx is None:
                from scipy import sparse
                bm = nat.bm25
                weights = bm.weights.tocsc()
                counts = sparse.load_npz(
                    os.path.join(NATIVE_DIR, "bm25_counts.npz")).tocsr()
                df = np.diff(weights.indptr)  # CSC: non-zeros per column
                n_docs = weights.shape[0]
                terms = [""] * len(bm.vocab)
                for term, col in bm.vocab.items():
                    terms[col] = term
                lengths = np.asarray(counts.sum(axis=1),
                                     dtype=np.float64).ravel()
                self.mx = {
                    "csr": weights.tocsr(), "counts": counts, "df": df,
                    "idf": np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)),
                    "lengths": lengths, "avgdl": float(lengths.mean()),
                    "terms": terms,
                    "bytes": os.path.getsize(
                        os.path.join(NATIVE_DIR, "bm25_weights.npz")),
                }
            return self.mx

    def matrix_row(self, row: int) -> Dict[str, Any]:
        """Every non-zero cell of one row of W, with tf, df and idf."""
        mx = self.matrix_parts()
        nat = self.get_native()
        csr, counts = mx["csr"], mx["counts"]
        n_docs, n_terms = csr.shape
        if not 0 <= row < n_docs:
            raise BadRequest(f"row must be between 0 and {n_docs - 1}")
        a, b = csr.indptr[row], csr.indptr[row + 1]
        cols, values = csr.indices[a:b], csr.data[a:b]
        c0, c1 = counts.indptr[row], counts.indptr[row + 1]
        tf = dict(zip(counts.indices[c0:c1].tolist(),
                      counts.data[c0:c1].tolist()))
        order = np.argsort(-values)
        cells = [{"term": mx["terms"][int(cols[j])], "col": int(cols[j]),
                  "tf": int(tf.get(int(cols[j]), 0)),
                  "df": int(mx["df"][cols[j]]),
                  "idf": round(float(mx["idf"][cols[j]]), 4),
                  "w": round(float(values[j]), 4)} for j in order[:80]]
        chunk = nat.chunks[row]
        try:
            full = self.text_of(chunk.file_path)
        except OSError:
            full = ""
        text = full[chunk.first_character_index:chunk.last_character_index]
        searchable = NativeIndexer.searchable_text(chunk, full)
        words = [(word, tokenizer.tokenize(word))
                 for word in tokenizer.WORD_RE.findall(searchable)]
        return {"row": row, "shape": [n_docs, n_terms], "nnz": int(csr.nnz),
                "density": csr.nnz / float(n_docs * n_terms),
                "dense_mb": n_docs * n_terms * 4 / 2 ** 20,
                "sparse_mb": mx["bytes"] / 2 ** 20,
                "k1": nat.bm25.k1, "b": nat.bm25.b,
                "avgdl": round(mx["avgdl"], 2),
                "length": int(mx["lengths"][row]), "n_terms": int(b - a),
                "chunk": {**chunk.to_source().model_dump(),
                          "kind": chunk.kind, "context": chunk.context},
                "text": text[:PREVIEW_CHARS],
                "truncated": len(text) > PREVIEW_CHARS, "cells": cells,
                "words": words[:MAX_WORDS],
                "n_words": len(words)}

    # ---- vocabulary and files on disk --------------------------------
    def vocab(self, text: str, prefix: str) -> Dict[str, Any]:
        """One term of bm25_vocab.json and its whole column of W."""
        mx = self.matrix_parts()
        nat = self.get_native()
        vocab = nat.bm25.vocab
        with self.lock:
            if "sorted" not in mx:
                mx["sorted"] = sorted(vocab)
                mx["counts_csc"] = mx["counts"].tocsc()
        raw = text.strip().lower()
        tokens = tokenizer.tokenize(text) if text.strip() else []
        term = raw if raw in vocab else (tokens[0] if tokens else "")
        prefix = prefix or term or raw
        start = bisect.bisect_left(mx["sorted"], prefix.lower())
        matches = []
        for word in mx["sorted"][start:start + 40]:
            if not word.startswith(prefix.lower()):
                break
            matches.append({"term": word, "col": vocab[word],
                            "df": int(mx["df"][vocab[word]])})
        result: Dict[str, Any] = {
            "query": text, "tokens": tokens, "term": term,
            "found": term in vocab, "n_terms": len(vocab),
            "n_docs": int(mx["csr"].shape[0]), "matches": matches,
            "prefix": prefix}
        if term not in vocab:
            return result
        col = vocab[term]
        weights = nat.bm25.weights
        a, b = weights.indptr[col], weights.indptr[col + 1]
        rows, values = weights.indices[a:b], weights.data[a:b]
        counts = mx["counts_csc"]
        c0, c1 = counts.indptr[col], counts.indptr[col + 1]
        tf = dict(zip(counts.indices[c0:c1].tolist(),
                      counts.data[c0:c1].tolist()))
        order = np.argsort(-values)[:VOCAB_ROWS]
        cells = []
        for j in order:
            row = int(rows[j])
            chunk = nat.chunks[row]
            cells.append({"row": row, "w": round(float(values[j]), 4),
                          "tf": int(tf.get(row, 0)),
                          "length": int(mx["lengths"][row]),
                          "file_path": chunk.file_path, "kind": chunk.kind,
                          "context": chunk.context[:80]})
        result.update(col=col, df=int(b - a),
                      idf=round(float(mx["idf"][col]), 4),
                      json_entry=json.dumps({term: col}), cells=cells,
                      sorted_pos=bisect.bisect_left(mx["sorted"], term))
        return result

    def files(self) -> Dict[str, Any]:
        """What the native index really is on disk, file by file."""
        with self.lock:
            if self.disk is not None:
                return self.disk
        items = []
        for name in sorted(os.listdir(NATIVE_DIR)):
            path = os.path.join(NATIVE_DIR, name)
            if not os.path.isfile(path):
                continue
            item: Dict[str, Any] = {"name": name, "path": path,
                                    "bytes": os.path.getsize(path)}
            try:
                item.update(self._file_detail(path))
            except (OSError, ValueError) as err:
                item["error"] = str(err)
            items.append(item)
        result = {"dir": NATIVE_DIR, "files": items,
                  "total": sum(i["bytes"] for i in items)}
        with self.lock:
            self.disk = result
        return result

    @staticmethod
    def _file_detail(path: str) -> Dict[str, Any]:
        """Format and the first values of one index file."""
        def head(values: Any) -> List[Any]:
            flat = np.asarray(values).ravel()[:FILE_HEAD]
            return [v.decode() if isinstance(v, bytes) else
                    float(f"{float(v):.4g}") if isinstance(v, np.floating)
                    else v.item() if hasattr(v, "item") else v
                    for v in flat]
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as npz:
                arrays = [{"name": key, "dtype": str(npz[key].dtype),
                           "shape": list(npz[key].shape),
                           "head": head(npz[key])} for key in npz.files]
            return {"format": "npz", "arrays": arrays}
        if path.endswith(".npy"):
            arr = np.load(path, mmap_mode="r")
            return {"format": "npy", "dtype": str(arr.dtype),
                    "shape": list(arr.shape),
                    "head": head(arr[0] if arr.ndim > 1 else arr)}
        if path.endswith(".jsonl"):
            with open(path, encoding="utf-8") as handle:
                lines = [handle.readline().rstrip("\n") for _ in range(3)]
                n_lines = 3 + sum(1 for _ in handle)
            return {"format": "jsonl", "lines": n_lines, "head": lines}
        if path.endswith(".json"):
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                first = dict(list(data.items())[:FILE_HEAD])
                return {"format": "json", "keys": len(data),
                        "head": json.dumps(first, indent=1)}
            return {"format": "json", "head": json.dumps(data)[:600]}
        return {"format": os.path.splitext(path)[1].lstrip(".") or "file"}

    # ---- retrieval -----------------------------------------------------
    def _hit(self, path: str, first: int, last: int, score: float,
             terms: Dict[str, float], kind: str, context: str
             ) -> Dict[str, Any]:
        """One retrieved chunk with a preview of its text."""
        try:
            body = self.text_of(path)[first:last]
        except OSError:
            body = ""
        return {
            "file_path": path, "first_character_index": first,
            "last_character_index": last, "chars": last - first,
            "score": round(float(score), 4),
            "terms": {t: round(v, 4) for t, v in
                      sorted(terms.items(), key=lambda kv: -kv[1])},
            "kind": kind, "context": context,
            "preview": body[:PREVIEW_CHARS],
            "truncated": len(body) > PREVIEW_CHARS,
        }

    def retrieve_native(self, tokens: List[str], query: str, k: int
                        ) -> Dict[str, Any]:
        """Native BM25: one sparse column sum per query term."""
        nat = self.get_native()
        bm = nat.bm25
        start = time.perf_counter()
        scores = bm.score(tokens)
        ranking = nat.ranking(query, k)
        elapsed = ms(start)
        cols = {t: bm.vocab[t] for t in set(tokens) if t in bm.vocab}
        term_info = []
        for t in sorted(set(tokens)):
            col = cols.get(t)
            df = int(bm.weights[:, col].nnz) if col is not None else 0
            term_info.append({"term": t, "in_vocab": col is not None,
                              "df": df})
        hits = []
        for i in ranking:
            chunk = nat.chunks[i]
            terms = {t: float(bm.weights[i, c]) for t, c in cols.items()}
            terms = {t: v for t, v in terms.items() if v > 0}
            hits.append(self._hit(chunk.file_path,
                                  chunk.first_character_index,
                                  chunk.last_character_index,
                                  float(scores[i]), terms, chunk.kind,
                                  chunk.context))
            hits[-1]["row"] = int(i)
        return {"scored": int(scores.shape[0]),
                "matched": int(np.count_nonzero(scores > 0)),
                "terms": term_info, "hits": hits, "score_ms": elapsed,
                "how": "scores = W[:, query_terms].sum(axis=1)  "
                       "(W: precomputed BM25 weights, scipy CSC)"}

    def retrieve_langchain(self, tokens: List[str], query: str, k: int
                           ) -> Dict[str, Any]:
        """LangChain BM25Retriever.invoke (rank_bm25 BM25Okapi inside)."""
        lc = self.get_langchain()
        vec = lc.vectorizer
        with self.lock:
            start = time.perf_counter()
            lc.k = k
            docs = lc.invoke(query)
            elapsed = ms(start)
        scores = np.asarray(vec.get_scores(tokens), dtype=np.float64)
        term_info = []
        for t in sorted(set(tokens)):
            df = sum(1 for freqs in vec.doc_freqs if t in freqs)
            term_info.append({"term": t, "in_vocab": t in vec.idf,
                              "df": df})
        hits = []
        for doc in docs:
            i = self.lc_pos.get(id(doc), -1)
            terms: Dict[str, float] = {}
            if i >= 0:
                dl = vec.doc_len[i]
                for t in set(tokens):
                    f = vec.doc_freqs[i].get(t, 0)
                    if f and t in vec.idf:
                        norm = vec.k1 * (1 - vec.b + vec.b * dl / vec.avgdl)
                        terms[t] = float(vec.idf[t] * f * (vec.k1 + 1)
                                         / (f + norm))
            meta = doc.metadata
            hits.append(self._hit(meta["file_path"],
                                  meta["first_character_index"],
                                  meta["last_character_index"],
                                  float(scores[i]) if i >= 0 else 0.0,
                                  terms, "splitter", ""))
        return {"scored": len(vec.doc_freqs),
                "matched": int(np.count_nonzero(scores > 0)),
                "terms": term_info, "hits": hits, "score_ms": elapsed,
                "how": "BM25Retriever.invoke(query) -> "
                       "BM25Okapi.get_scores(tokenize(query)), "
                       "Python loop over every document"}

    def embed_query(self, query: str) -> Dict[str, Any]:
        """MiniLM: word pieces and the 384-d query vector."""
        qd = self.get_qdrant()
        start = time.perf_counter()
        vector = qd.embedder.encode([query])[0]
        elapsed = ms(start)
        pieces = qd.embedder.tokenizer.tokenize(query)
        return {"pieces": pieces, "dim": int(vector.shape[0]),
                "head": [round(float(v), 4) for v in vector[:48]],
                "norm": round(float(np.linalg.norm(vector)), 4),
                "model": qd.meta.get("model"), "ms": elapsed,
                "vector": vector}

    def retrieve_qdrant(self, vector: Any, k: int) -> Dict[str, Any]:
        """Nearest points by cosine in the embedded Qdrant collection."""
        qd = self.get_qdrant()
        start = time.perf_counter()
        points = qd.client.query_points(COLLECTION, query=vector.tolist(),
                                        limit=k, with_payload=True).points
        elapsed = ms(start)
        total = qd.client.count(COLLECTION).count
        hits = []
        for point in points:
            pay = dict(point.payload or {})
            hits.append(self._hit(pay["file_path"],
                                  pay["first_character_index"],
                                  pay["last_character_index"],
                                  float(point.score), {},
                                  str(pay.get("kind", "")),
                                  str(pay.get("context", ""))))
        return {"scored": int(total), "matched": int(total), "terms": [],
                "hits": hits, "score_ms": elapsed,
                "how": "client.query_points(collection, query=vector, "
                       "limit=k): cosine against every stored vector"}

    # ---- augment -------------------------------------------------------
    def augment(self, query: str, hits: List[Dict[str, Any]]
                ) -> Dict[str, Any]:
        """Mirror of AnswerGenerator.build_context, with per-block data."""
        start = time.perf_counter()
        try:
            tok = self.get_tokenizer()
        except Exception as err:  # noqa: BLE001 - no tokenizer available
            return {"error": f"Qwen tokenizer unavailable: {err}"}
        blocks = []
        parts: List[str] = []
        used = 0
        stopped = False
        for number, hit in enumerate(hits, start=1):
            snippet = self.text_of(hit["file_path"])[
                hit["first_character_index"]:hit["last_character_index"]]
            block = f"[{number}] {hit['file_path']}\n{snippet.strip()}\n"
            n_tokens = len(tok.encode(block))
            kept = not stopped and used + n_tokens <= MAX_CONTEXT_TOKENS
            if kept:
                parts.append(block)
                used += n_tokens
            elif not stopped and not parts:
                ids = tok.encode(block)
                parts.append(tok.decode(ids[:MAX_CONTEXT_TOKENS]))
                used = MAX_CONTEXT_TOKENS
                kept = True
                stopped = True
            else:
                stopped = True
            blocks.append({"n": number, "file_path": hit["file_path"],
                           "tokens": n_tokens, "kept": kept})
        context = "\n".join(parts)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        return {"budget": MAX_CONTEXT_TOKENS, "used": used,
                "blocks": blocks, "prompt": prompt,
                "prompt_tokens": len(tok.encode(prompt)),
                "system_prompt": SYSTEM_PROMPT, "ms": ms(start)}

    # ---- full trace ----------------------------------------------------
    def trace(self, query: str, k: int, backend: str) -> Dict[str, Any]:
        """Run tokenize -> retrieve -> augment and time every stage."""
        total = time.perf_counter()
        start = time.perf_counter()
        words = tokenizer.WORD_RE.findall(query)
        tokens = tokenizer.tokenize(query)
        tok_stage = {"words": words, "tokens": tokens, "ms": ms(start)}
        start = time.perf_counter()
        if backend == "qdrant":
            emb = self.embed_query(query)
            vector = emb.pop("vector")
            tok_stage["embed"] = emb
            tok_stage["ms"] = emb["ms"]
            start = time.perf_counter()
            ret = self.retrieve_qdrant(vector, k)
        elif backend == "langchain":
            ret = self.retrieve_langchain(tokens, query, k)
        else:
            ret = self.retrieve_native(tokens, query, k)
        ret["ms"] = ms(start)
        aug = self.augment(query, ret["hits"])
        sources = [{key: h[key] for key in
                    ("file_path", "first_character_index",
                     "last_character_index")} for h in ret["hits"]]
        return {"query": query, "k": k, "backend": backend,
                "tokenize": tok_stage, "retrieve": ret, "augment": aug,
                "sources": sources, "total_ms": ms(total)}

    def generate(self, query: str, sources: List[MinimalSource]
                 ) -> Dict[str, Any]:
        """Answer with the project's generator, exactly like the CLI."""
        load_start = time.perf_counter()
        gen = self.get_generator()
        load_ms = ms(load_start)
        start = time.perf_counter()
        with self.gen_lock:
            answer = (gen.answer(query, sources) if sources
                      else NO_CONTEXT_ANSWER)
        result = MinimalAnswer(question_id="query", question=query,
                               retrieved_sources=sources, answer=answer)
        return {"answer": answer, "ms": ms(start), "load_ms": load_ms,
                "device": getattr(gen, "device", "cpu"),
                "model": DEFAULT_MODEL, "output": result.model_dump()}


ENGINE = Engine()
STATIC_DIR = ""  # set by --static: serve the UI without nginx


def check_query(value: Any) -> str:
    """Validate the question."""
    if not isinstance(value, str) or not value.strip():
        raise BadRequest("query must be a non-empty string")
    if len(value) > 500:
        raise BadRequest("query is longer than 500 characters")
    return value.strip()


def check_k(value: Any) -> int:
    """Validate k."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise BadRequest("k must be an integer")
    if not 1 <= value <= 20:
        raise BadRequest("k must be between 1 and 20")
    return value


def int_arg(args: Dict[str, List[str]], name: str, default: int,
            low: int, high: int) -> int:
    """Validate an integer query-string argument."""
    raw = args.get(name, [str(default)])[0]
    try:
        value = int(raw)
    except ValueError as err:
        raise BadRequest(f"{name} must be an integer") from err
    if not low <= value <= high:
        raise BadRequest(f"{name} must be between {low} and {high}")
    return value


def check_sources(value: Any) -> List[MinimalSource]:
    """Validate client-sent sources: only spans inside data/raw."""
    if not isinstance(value, list) or len(value) > 20:
        raise BadRequest("sources must be a list of at most 20 items")
    out = []
    for item in value:
        try:
            src = MinimalSource(**item)
        except Exception as err:  # noqa: BLE001
            raise BadRequest(f"invalid source: {err}") from err
        path = os.path.normpath(src.file_path)
        if not path.startswith("data/raw/") or ".." in path.split("/"):
            raise BadRequest(f"source outside data/raw: {src.file_path}")
        out.append(src)
    return out


class Handler(BaseHTTPRequestHandler):
    """JSON request handler."""

    server_version = "RAGTrace/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        """One log line per request."""
        print(f"[trace] {format % args}", file=sys.stderr, flush=True)

    def do_GET(self) -> None:
        """Handle GET."""
        self._dispatch("GET")

    def do_POST(self) -> None:
        """Handle POST."""
        self._dispatch("POST")

    def _body(self) -> Dict[str, Any]:
        """Decode the JSON body."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as err:
            raise BadRequest("bad Content-Length") from err
        if length < 0 or length > 1 << 20:
            raise BadRequest("body too large")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            raise BadRequest(f"malformed JSON: {err}") from err
        if not isinstance(body, dict):
            raise BadRequest("JSON body must be an object")
        return body

    def _route(self, method: str, path: str, query: str
               ) -> Tuple[int, Any]:
        """Return (status, payload)."""
        if (STATIC_DIR and method == "GET"
                and path in ("/", "/index.html")):
            return 200, None
        if path == "/api/health" and method == "GET":
            return 200, {"status": "ok", "cwd": os.getcwd(),
                         "loaded": {"native": ENGINE.native is not None,
                                    "langchain": ENGINE.lc is not None,
                                    "qdrant": ENGINE.qd is not None,
                                    "qwen3": ENGINE.generator is not None},
                         "load_ms": ENGINE.load_times}
        if path == "/api/inside" and method == "GET":
            with ENGINE.lock:
                if ENGINE.inside is None:
                    ENGINE.inside = describe_inside()
            return 200, ENGINE.inside
        if path == "/api/stats" and method == "GET":
            return 200, ENGINE.stats()
        if path == "/api/chunks" and method == "GET":
            args = parse_qs(query)
            backend = args.get("backend", ["native"])[0]
            if backend not in BACKENDS:
                raise BadRequest(f"backend must be one of {BACKENDS}")
            return 200, ENGINE.chunks_page(
                backend, int_arg(args, "offset", 0, 0, 10 ** 7),
                int_arg(args, "limit", 25, 1, 100),
                args.get("q", [""])[0].strip()[:200])
        if path == "/api/matrix" and method == "GET":
            args = parse_qs(query)
            return 200, ENGINE.matrix_row(
                int_arg(args, "row", 0, 0, 10 ** 7))
        if path == "/api/vocab" and method == "GET":
            args = parse_qs(query)
            return 200, ENGINE.vocab(args.get("term", [""])[0][:100],
                                     args.get("prefix", [""])[0][:60])
        if path == "/api/files" and method == "GET":
            return 200, ENGINE.files()
        if path == "/api/trace" and method == "POST":
            body = self._body()
            backend = body.get("backend", "native")
            if backend not in BACKENDS:
                raise BadRequest(f"backend must be one of {BACKENDS}")
            return 200, ENGINE.trace(check_query(body.get("query")),
                                     check_k(body.get("k", 5)), backend)
        if path == "/api/generate" and method == "POST":
            body = self._body()
            return 200, ENGINE.generate(check_query(body.get("query")),
                                        check_sources(body.get("sources",
                                                               [])))
        if path == "/api/mcp" and method == "POST":
            body = self._body()
            return 200, record_session(check_query(body.get("query")),
                                       check_k(body.get("k", 3)),
                                       os.getcwd())
        return 404, {"error": f"no route {method} {path}"}

    def _dispatch(self, method: str) -> None:
        """Run a route and always answer JSON."""
        try:
            path, _, query = self.path.partition("?")
            status, payload = self._route(method, path, query)
        except BadRequest as err:
            status, payload = 400, {"error": str(err)}
        except Exception as err:  # noqa: BLE001 - keep serving
            status, payload = 500, {"error": f"{type(err).__name__}: {err}"}
        ctype = "application/json"
        if payload is None:
            with open(os.path.join(STATIC_DIR, "index.html"), "rb") as page:
                data = page.read()
            ctype = "text/html; charset=utf-8"
        else:
            data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def warm_up() -> None:
    """Load both indices and the tokenizer in the background."""
    for step in (ENGINE.get_native, ENGINE.get_langchain,
                 ENGINE.get_tokenizer):
        try:
            step()
        except Exception as err:  # noqa: BLE001
            print(f"[trace] warm-up {step.__name__} failed: {err}",
                  file=sys.stderr, flush=True)
    print(f"[trace] warm-up done: {ENGINE.load_times}", file=sys.stderr,
          flush=True)


def main() -> None:
    """Parse arguments and serve forever."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--static", default="",
                        help="directory with index.html (no-Docker mode)")
    args = parser.parse_args()
    global STATIC_DIR
    STATIC_DIR = os.path.abspath(args.static) if args.static else ""
    threading.Thread(target=warm_up, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"[trace] serving on {args.host}:{args.port} from {os.getcwd()}",
          file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
