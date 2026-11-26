"""Native BM25 retrieval: score every chunk, keep the k best."""

from typing import List, Optional


from ..models.chunk import Chunk
from ..repositories.bm25_repository import bm25_repository
from ..repositories.chunk_repository import chunk_repository
from ..schemas.source import MinimalSource
from .ranker import ranker
from .search_backend import is_blank
from .tokenizer import tokenizer


class LexicalRetriever:
    """Native BM25: score every chunk, keep the k best (score > 0)."""

    def __init__(self, index_dir: str,
                 chunks: Optional[List[Chunk]] = None) -> None:
        """Load chunks and BM25 weights from ``index_dir``.

        Raises:
            FileNotFoundError: If the index has not been built yet.
            ValueError: If the index files do not match each other.
        """
        self.chunks: List[Chunk] = (chunks if chunks is not None
                                    else chunk_repository.load(index_dir))
        self.bm25 = bm25_repository.load(index_dir)
        if self.bm25.n_documents != len(self.chunks):
            raise ValueError("BM25 matrix and chunks do not match: re-run "
                             "the 'index' command")

    def ranking(self, query: str, n: int) -> List[int]:
        """Chunk indices of the ``n`` best BM25 matches."""
        scores = self.bm25.score(tokenizer.tokenize(query))
        return ranker.top_indices(scores, n, positive_only=True)

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """Return the ``k`` best source spans for ``query``."""
        if is_blank(query, k):
            return []
        return [self.chunks[i].to_source() for i in self.ranking(query, k)]
