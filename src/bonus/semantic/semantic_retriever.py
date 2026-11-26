"""Semantic retrieval: cosine between the question and every chunk vector."""

from typing import List, Optional

import numpy as np

from ...models.chunk import Chunk
from ...repositories.chunk_repository import chunk_repository
from ...repositories.meta_repository import meta_repository
from ...schemas.source import MinimalSource
from ...services.ranker import ranker
from ...services.search_backend import is_blank
from .embedder import embedder_provider
from .embedding_repository import embedding_repository


class SemanticRetriever:
    """Cosine similarity between the question and every chunk vector."""

    def __init__(self, index_dir: str,
                 chunks: Optional[List[Chunk]] = None) -> None:
        """Load chunks, vectors and the encoder used to build them.

        Raises:
            FileNotFoundError: If the semantic index has not been built.
            ValueError: If the vectors belong to an older index.
        """
        meta = embedding_repository.read_meta(index_dir)
        if not embedding_repository.exists(index_dir) or not meta:
            raise FileNotFoundError(
                f"no semantic index found in {index_dir!r}: run "
                "'index --semantic' first")
        self.chunks = (chunks if chunks is not None
                       else chunk_repository.load(index_dir))
        self.vectors = embedding_repository.load_vectors(index_dir)
        if meta.get("build_id") != meta_repository.build_id(index_dir) \
                or self.vectors.shape[0] != len(self.chunks):
            raise ValueError("embeddings do not match the current index: "
                             "re-run 'index --semantic'")
        self.embedder = embedder_provider.create(str(meta["model"]))

    def scores(self, query: str) -> np.ndarray:
        """Cosine similarity between the query and every chunk."""
        query_vector = self.embedder.encode([query])[0]
        result: np.ndarray = self.vectors @ query_vector
        return result

    def ranking(self, query: str, n: int) -> List[int]:
        """Chunk indices of the ``n`` closest chunks, best first."""
        if not self.chunks:
            return []
        return ranker.top_indices(self.scores(query), n)

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """Return the ``k`` chunks closest in meaning to ``query``."""
        if is_blank(query, k):
            return []
        return [self.chunks[i].to_source() for i in self.ranking(query, k)]
