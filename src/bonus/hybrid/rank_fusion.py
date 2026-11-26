"""Reciprocal Rank Fusion: merge several rankings into one."""

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


class RankFusion:
    """Weighted Reciprocal Rank Fusion of ranked id lists."""

    rrf_k = 30

    def fuse(self, rankings: Sequence[Tuple[List[int], float]],
             rrf_k: int = rrf_k) -> List[int]:
        """Merge rankings with Reciprocal Rank Fusion.

            fused(d) = sum_r  weight_r / (rrf_k + rank_r(d))

        Args:
            rankings: ``(ranked ids best first, weight)`` per retriever.
            rrf_k: Smoothing constant; larger values flatten rank gaps.

        Returns:
            Every id of every ranking, best fused score first (ties are
            broken by the smaller id, for determinism).
        """
        fused: Dict[int, float] = defaultdict(float)
        for ranking, weight in rankings:
            for rank, item in enumerate(ranking, start=1):
                fused[item] += weight / (rrf_k + rank)
        return sorted(fused, key=lambda item: (-fused[item], item))


rank_fusion = RankFusion()
