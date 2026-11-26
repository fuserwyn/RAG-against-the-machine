"""Hybrid retrieval: BM25 and semantic rankings fused with RRF."""

from typing import List

from ...schemas.source import MinimalSource
from ...services.retriever import LexicalRetriever
from ...services.search_backend import is_blank
from ..semantic.semantic_retriever import SemanticRetriever
from .rank_fusion import rank_fusion


class HybridRetriever:
    """BM25 + semantic rankings combined with Reciprocal Rank Fusion.

    BM25 is unbounded and cosine is in [-1, 1], so the two are combined
    by rank, not by score. MiniLM alone is much weaker than BM25 on this
    corpus, so the semantic ranking gets a small weight: it mostly
    re-orders and completes the BM25 candidates (weights from a grid
    search on the public datasets, see the README).
    """

    LEXICAL_WEIGHT = 1.0
    SEMANTIC_WEIGHT = 0.2

    def __init__(self, index_dir: str, depth: int = 100,
                 lexical_weight: float = LEXICAL_WEIGHT,
                 semantic_weight: float = SEMANTIC_WEIGHT,
                 rrf_k: int = rank_fusion.rrf_k) -> None:
        """Load both indexes (they share the same chunks).

        Args:
            index_dir: Directory written by ``index --semantic``.
            depth: How many candidates each retriever contributes.
            lexical_weight: Weight of the BM25 ranking in the fusion.
            semantic_weight: Weight of the embedding ranking.
            rrf_k: Reciprocal Rank Fusion smoothing constant.
        """
        self.lexical = LexicalRetriever(index_dir)
        self.semantic = SemanticRetriever(index_dir,
                                          chunks=self.lexical.chunks)
        self.depth = depth
        self.weights = (lexical_weight, semantic_weight)
        self.rrf_k = rrf_k

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """Return the ``k`` best fused source spans for ``query``."""
        if is_blank(query, k):
            return []
        depth = max(self.depth, k)
        fused = rank_fusion.fuse([
            (self.lexical.ranking(query, depth), self.weights[0]),
            (self.semantic.ranking(query, depth), self.weights[1]),
        ], self.rrf_k)
        return [self.lexical.chunks[i].to_source() for i in fused[:k]]
