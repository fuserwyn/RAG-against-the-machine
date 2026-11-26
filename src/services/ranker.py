"""Turning a score per chunk into a top-k ranking."""

from typing import List

import numpy as np


class Ranker:
    """Top-k selection."""

    def top_indices(self, scores: np.ndarray, n: int,
                    positive_only: bool = False) -> List[int]:
        """Indices of the ``n`` highest scores, best first.

        ``argpartition`` finds the ``n`` best without sorting everything;
        only those ``n`` are then sorted. With ``positive_only``, indices
        whose score is not strictly positive are dropped.
        """
        n = min(n, int(scores.shape[0]))
        if n <= 0:
            return []
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [int(i) for i in top if not positive_only or scores[i] > 0]


ranker = Ranker()
