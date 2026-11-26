"""Semantic index (bonus 1): one MiniLM vector per native chunk.

Vectors are L2-normalized, so the cosine similarity between a question
and every chunk is a single matrix-vector product. They are stored as a
plain numpy matrix (``embeddings.npy``): at ~23k chunks no vector
database is needed. They are tied to the ``build_id`` of the lexical
index; after an incremental build only the changed chunks are embedded.
"""

from typing import Dict, List, Optional

import numpy as np

from ...config import settings
from ...models.chunk import Chunk
from ...models.index_build import IndexReport
from ...repositories.chunk_repository import ChunkRepository, chunk_repository
from ...repositories.corpus_repository import (CorpusRepository,
                                               corpus_repository)
from ...repositories.meta_repository import MetaRepository, meta_repository
from .embedder import EmbedderProvider, embedder_provider
from .embedding_repository import EmbeddingRepository, embedding_repository


class SemanticIndexer:
    """Embeds the chunks of an existing native index."""

    def __init__(self, chunks: ChunkRepository, corpus: CorpusRepository,
                 vectors: EmbeddingRepository, meta: MetaRepository,
                 embedders: EmbedderProvider) -> None:
        """Wire the repositories and the encoder provider."""
        self.chunks = chunks
        self.corpus = corpus
        self.vectors = vectors
        self.meta = meta
        self.embedders = embedders

    @staticmethod
    def embedding_text(chunk: Chunk, text: str) -> str:
        """Encoder input: context (class / headings) + chunk body."""
        body = chunk.body(text)
        return f"{chunk.context}\n{body}" if chunk.context else body

    def build(self, index_dir: str, report: Optional[IndexReport] = None,
              model_name: str = settings.embedding_model) -> int:
        """Embed the chunks of ``index_dir`` and save the vectors.

        Args:
            index_dir: Native index directory (``index`` must have run).
            report: Report of the lexical build that just ran. After an
                incremental build, vectors of unchanged chunks are reused.
            model_name: Hugging Face encoder id.

        Returns:
            The number of chunks embedded by this call.
        """
        chunks = self.chunks.load(index_dir)
        origin = [-1] * len(chunks)
        reuse = None
        if report is not None and len(report.row_origin) == len(chunks):
            origin = report.row_origin
            reuse = self._reusable_vectors(index_dir, report, model_name)
        todo = [i for i, row in enumerate(origin) if reuse is None or row < 0]
        fresh = None
        if todo:
            encoder = self.embedders.create(model_name)
            fresh = encoder.encode(self._texts(chunks, todo), progress=True)
        vectors = self._assemble(len(chunks), origin, reuse, todo, fresh)
        self.vectors.save(index_dir, vectors, model=model_name,
                          n_chunks=len(chunks), dim=int(vectors.shape[1]),
                          build_id=self.meta.build_id(index_dir))
        return len(todo)

    def _texts(self, chunks: List[Chunk], rows: List[int]) -> List[str]:
        """Encoder input of the selected chunks (each file read once)."""
        cache: Dict[str, str] = {}
        texts: List[str] = []
        for row in rows:
            chunk = chunks[row]
            if chunk.file_path not in cache:
                cache = {chunk.file_path:
                         self.corpus.read_text(chunk.file_path)}
            texts.append(self.embedding_text(chunk, cache[chunk.file_path]))
        return texts

    def _reusable_vectors(self, index_dir: str, report: IndexReport,
                          model_name: str) -> Optional[np.ndarray]:
        """Vectors of the previous build, if they can be reused."""
        meta = self.vectors.read_meta(index_dir)
        if report.previous_build_id is None \
                or meta.get("build_id") != report.previous_build_id \
                or meta.get("model") != model_name:
            return None
        try:
            vectors = self.vectors.load_vectors(index_dir)
        except (OSError, ValueError):
            return None
        if vectors.ndim != 2 or \
                max(report.row_origin, default=-1) >= vectors.shape[0]:
            return None
        return vectors

    @staticmethod
    def _assemble(n_chunks: int, origin: List[int],
                  reuse: Optional[np.ndarray], todo: List[int],
                  fresh: Optional[np.ndarray]) -> np.ndarray:
        """Put reused and fresh vectors at the row of their chunk."""
        if fresh is not None:
            dim = int(fresh.shape[1])
        else:
            dim = int(reuse.shape[1]) if reuse is not None else 0
        vectors = np.zeros((n_chunks, dim), dtype=np.float32)
        if reuse is not None:
            kept = [i for i, row in enumerate(origin) if row >= 0]
            vectors[kept] = reuse[[origin[i] for i in kept]]
        if fresh is not None:
            vectors[todo] = fresh
        return vectors


semantic_indexer = SemanticIndexer(
    chunks=chunk_repository, corpus=corpus_repository,
    vectors=embedding_repository, meta=meta_repository,
    embedders=embedder_provider)
