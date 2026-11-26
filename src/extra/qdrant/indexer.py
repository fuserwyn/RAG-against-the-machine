"""Qdrant ingestion: native chunks -> MiniLM vectors -> collection.

Chunking reuses the native AST / heading chunkers, so the three backends
differ only in retrieval. Each chunk is embedded with the same encoder as
the semantic bonus and stored as one point whose payload is its span.
"""

from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from ...bonus.semantic.embedder import EmbedderProvider, embedder_provider
from ...bonus.semantic.semantic_indexer import SemanticIndexer
from ...config import settings
from ...repositories.corpus_repository import (CorpusRepository,
                                               corpus_repository)
from ...repositories.meta_repository import MetaRepository, meta_repository
from ...services.chunkers import FileChunker, file_chunker
from .qdrant_repository import QdrantRepository, qdrant_repository


class QdrantIndexer:
    """Builds the Qdrant collection of a corpus."""

    def __init__(self, corpus: CorpusRepository, store: QdrantRepository,
                 meta: MetaRepository, chunker: FileChunker,
                 embedders: EmbedderProvider) -> None:
        """Wire the repositories and services the build relies on."""
        self.corpus = corpus
        self.store = store
        self.meta = meta
        self.chunker = chunker
        self.embedders = embedders

    def build(self, raw_dir: str, index_dir: str, max_chunk_size: int,
              merge_ratio: float = settings.merge_ratio,
              model_name: str = settings.embedding_model) -> int:
        """Chunk, embed and upsert the corpus into a fresh collection.

        Returns:
            The number of points stored.
        """
        payloads, texts = self._chunk_corpus(raw_dir, max_chunk_size,
                                             merge_ratio)
        if not payloads:
            raise FileNotFoundError(f"no chunk produced from {raw_dir!r}")
        vectors = self.embedders.create(model_name).encode(texts,
                                                           progress=True)
        self.store.save(index_dir, vectors, payloads)
        self.meta.write(index_dir, raw_dir=raw_dir,
                        max_chunk_size=max_chunk_size,
                        merge_ratio=merge_ratio, n_chunks=len(payloads),
                        model=model_name, dim=int(vectors.shape[1]),
                        distance="cosine", collection=self.store.collection)
        return len(payloads)

    def _chunk_corpus(self, raw_dir: str, max_chunk_size: int,
                      merge_ratio: float
                      ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Payload and encoder input of every chunk of the corpus."""
        files = self.corpus.discover(raw_dir)
        if not files:
            raise FileNotFoundError(
                f"no indexable files found in {raw_dir!r}")
        merge_target = max(1, int(max_chunk_size * merge_ratio))
        payloads: List[Dict[str, Any]] = []
        texts: List[str] = []
        for path in tqdm(files, desc="Chunking files (qdrant)", unit="file"):
            try:
                text = self.corpus.read_text(path)
            except OSError as err:
                tqdm.write(f"Skipping {path}: {err}")
                continue
            for chunk in self.chunker.chunk(text, path, max_chunk_size,
                                            merge_target):
                payloads.append({**chunk.to_source().model_dump(),
                                 "kind": chunk.kind,
                                 "context": chunk.context})
                texts.append(SemanticIndexer.embedding_text(chunk, text))
        return payloads, texts


qdrant_indexer = QdrantIndexer(
    corpus=corpus_repository, store=qdrant_repository, meta=meta_repository,
    chunker=file_chunker, embedders=embedder_provider)
