"""Native ingestion: chunk every corpus file and build the BM25 index.

The build goes through four steps:

    1. discover   list the corpus files               (CorpusRepository)
    2. collect    chunk + tokenize each file          (FileChunker, Tokenizer)
    3. weigh      term counts -> BM25 weights         (BM25Index)
    4. save       chunks, matrix, manifest, meta      (repositories)

An incremental build (bonus 3) compares the SHA-1 of every file with the
previous build: unchanged files keep their chunks and term counts, only
added or modified files are chunked again, removed files are dropped.
The weights are then recomputed from the merged counts, so the result is
exactly the index a full rebuild would produce.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse
from tqdm import tqdm

from ..bonus.incremental.manifest_repository import (ManifestRepository,
                                                     manifest_repository)
from ..bonus.incremental.previous_build import PreviousBuild
from ..bonus.incremental.previous_build_loader import (PreviousBuildLoader,
                                                       previous_build_loader)
from ..models.bm25_index import BM25Index
from ..models.chunk import Chunk
from ..models.index_build import BuildSettings, IndexReport
from ..repositories.bm25_repository import BM25Repository, bm25_repository
from ..repositories.chunk_repository import ChunkRepository, chunk_repository
from ..repositories.corpus_repository import (CorpusRepository,
                                              corpus_repository)
from ..repositories.meta_repository import MetaRepository, meta_repository
from .chunkers import FileChunker, file_chunker
from .tokenizer import Tokenizer, tokenizer


@dataclass
class _BuildState:
    """Everything collected while walking the corpus.

    ``order[i]`` says where the term counts of chunk ``i`` come from: a
    row of the previous matrix (reused file) or ``n_old + j`` for the
    j-th freshly tokenized chunk.
    """

    previous: Optional[PreviousBuild]
    chunks: List[Chunk] = field(default_factory=list)
    order: List[int] = field(default_factory=list)
    new_documents: List[List[str]] = field(default_factory=list)
    digests: Dict[str, str] = field(default_factory=dict)
    indexed: List[str] = field(default_factory=list)
    reused: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Pre-compute the lookups of the previous build."""
        self.n_old = len(self.previous.chunks) if self.previous else 0
        self.old_digests = self.previous.digests if self.previous else {}
        self.old_rows = self.previous.rows_by_file() if self.previous else {}

    def is_unchanged(self, path: str, digest: str) -> bool:
        """Whether the previous build already indexed this exact file."""
        return self.previous is not None \
            and self.old_digests.get(path) == digest

    def reuse_file(self, path: str) -> None:
        """Keep the chunks and counts of an unchanged file."""
        assert self.previous is not None
        rows = self.old_rows.get(path, [])
        self.chunks.extend(self.previous.chunks[row] for row in rows)
        self.order.extend(rows)
        self.reused.append(path)

    def add_chunk(self, chunk: Chunk, tokens: List[str]) -> None:
        """Register a freshly chunked and tokenized piece."""
        self.chunks.append(chunk)
        self.order.append(self.n_old + len(self.new_documents))
        self.new_documents.append(tokens)

    def report(self, build_id: str) -> IndexReport:
        """Summary of the run."""
        return IndexReport(
            n_files=len(self.digests), n_chunks=len(self.chunks),
            build_id=build_id, incremental=self.previous is not None,
            previous_build_id=(self.previous.build_id if self.previous
                               else None),
            indexed_files=self.indexed, reused_files=self.reused,
            removed_files=sorted(set(self.old_digests) - set(self.digests)),
            row_origin=[row if row < self.n_old else -1
                        for row in self.order])


class NativeIndexer:
    """Builds (or updates) the native BM25 index of a corpus."""

    def __init__(self, corpus: CorpusRepository, chunks: ChunkRepository,
                 bm25: BM25Repository, manifest: ManifestRepository,
                 meta: MetaRepository, chunker: FileChunker,
                 text_tokenizer: Tokenizer,
                 history: PreviousBuildLoader) -> None:
        """Wire the repositories and services the build relies on."""
        self.corpus = corpus
        self.chunks = chunks
        self.bm25 = bm25
        self.manifest = manifest
        self.meta = meta
        self.chunker = chunker
        self.tokenizer = text_tokenizer
        self.history = history

    def build(self, raw_dir: str, index_dir: str, max_chunk_size: int,
              merge_ratio: float = 0.5,
              incremental: bool = False) -> IndexReport:
        """Chunk the corpus and persist chunks + BM25 index.

        Args:
            raw_dir: Directory holding the raw corpus.
            index_dir: Output directory for the index files.
            max_chunk_size: Hard upper bound on chunk length (characters).
            merge_ratio: Small pieces are merged up to
                ``max_chunk_size * merge_ratio`` characters.
            incremental: Reuse the previous build for unchanged files.
                Falls back to a full build when there is no compatible one.

        Returns:
            A report of what was (re)indexed.
        """
        config = BuildSettings(raw_dir=raw_dir,
                               max_chunk_size=max_chunk_size,
                               merge_ratio=merge_ratio)
        files = self._discover(raw_dir)
        previous = self.history.load(index_dir, config) if incremental \
            else None
        state = _BuildState(previous)
        for path in tqdm(files, desc="Chunking files", unit="file"):
            self._collect_file(state, path, config)
        index = self._weigh(state)
        build_id = self._save(index_dir, config, state, index)
        return state.report(build_id)

    @staticmethod
    def searchable_text(chunk: Chunk, text: str) -> str:
        """The text that is actually tokenized for one chunk.

        The file path and the chunk context (enclosing class / headings)
        are added to the chunk body: they often hold the exact words a
        question uses ("lora", "fused moe", ...).
        """
        path_words = chunk.file_path.replace("/", " ").replace(".", " ")
        return f"{path_words} {chunk.context}\n{chunk.body(text)}"

    # ---- step 1 -------------------------------------------------------
    def _discover(self, raw_dir: str) -> List[str]:
        """The files to index (at least one)."""
        files = self.corpus.discover(raw_dir)
        if not files:
            raise FileNotFoundError(
                f"no indexable files found in {raw_dir!r}")
        return files

    # ---- step 2 -------------------------------------------------------
    def _collect_file(self, state: _BuildState, path: str,
                      config: BuildSettings) -> None:
        """Reuse an unchanged file, or chunk and tokenize it."""
        try:
            text, digest = self.corpus.read_with_digest(path)
        except OSError as err:
            tqdm.write(f"Skipping {path}: {err}")
            return
        state.digests[path] = digest
        if state.is_unchanged(path, digest):
            state.reuse_file(path)
            return
        state.indexed.append(path)
        for chunk in self.chunker.chunk(text, path, config.max_chunk_size,
                                        config.merge_target):
            tokens = self.tokenizer.tokenize(self.searchable_text(chunk,
                                                                  text))
            state.add_chunk(chunk, tokens)

    # ---- step 3 -------------------------------------------------------
    def _weigh(self, state: _BuildState) -> BM25Index:
        """Count the terms of the new chunks and recompute all weights."""
        index = state.previous.bm25 if state.previous else BM25Index()
        new_counts = index.count_matrix(state.new_documents)
        index.set_counts(self._merge_counts(index, state, new_counts))
        return index

    @staticmethod
    def _merge_counts(index: BM25Index, state: _BuildState,
                      new_counts: sparse.csr_matrix) -> sparse.csr_matrix:
        """Stack reused and new count rows, then put them in corpus order."""
        n_terms = len(index.vocab)
        if state.previous is None:
            stacked = new_counts
        else:
            old = state.previous.bm25.counts
            old = sparse.csr_matrix((old.data, old.indices, old.indptr),
                                    shape=(old.shape[0], n_terms))
            stacked = sparse.vstack([old, new_counts], format="csr")
        if not state.order:
            return sparse.csr_matrix((0, n_terms), dtype=np.int32)
        return stacked[np.asarray(state.order, dtype=np.int64)]

    # ---- step 4 -------------------------------------------------------
    def _save(self, index_dir: str, config: BuildSettings,
              state: _BuildState, index: BM25Index) -> str:
        """Write every index file; return the new build id."""
        self.bm25.save(index, index_dir)
        self.chunks.save(index_dir, state.chunks)
        self.manifest.save(index_dir, state.digests)
        return self.meta.write(index_dir, **config.model_dump(),
                               n_files=len(state.digests),
                               n_chunks=len(state.chunks))


native_indexer = NativeIndexer(
    corpus=corpus_repository, chunks=chunk_repository, bm25=bm25_repository,
    manifest=manifest_repository, meta=meta_repository,
    chunker=file_chunker, text_tokenizer=tokenizer,
    history=previous_build_loader)
