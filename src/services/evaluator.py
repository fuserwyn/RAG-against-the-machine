"""Recall@k, for self-evaluation only (the official grader is the referee).

A reference source counts as found when a retrieved source is in the same
file and their spans overlap with an IoU of at least 0.05.
"""

from typing import Dict, List, Sequence, Tuple

from ..config import settings
from ..models.evaluation_report import EvaluationReport
from ..schemas.question import AnsweredQuestion, RagDataset
from ..schemas.search import StudentSearchResults
from ..schemas.source import MinimalSource

Pair = Tuple[List[MinimalSource], List[MinimalSource]]   # (got, expected)


class Evaluator:
    """Compares search results with the reference sources."""

    IOU_THRESHOLD = 0.05

    def iou(self, a: MinimalSource, b: MinimalSource) -> float:
        """Intersection over union of two spans (0 if other file)."""
        if a.file_path != b.file_path:
            return 0.0
        inter = min(a.last_character_index, b.last_character_index) \
            - max(a.first_character_index, b.first_character_index)
        if inter <= 0:
            return 0.0
        union = max(a.last_character_index, b.last_character_index) \
            - min(a.first_character_index, b.first_character_index)
        return inter / union if union > 0 else 0.0

    def recall_at_k(self, retrieved: Sequence[MinimalSource],
                    expected: Sequence[MinimalSource], k: int) -> float:
        """Share of the expected sources found among the top-k retrieved."""
        if not expected:
            return 1.0
        top = retrieved[:k]
        found = sum(1 for ref in expected
                    if any(self.iou(ref, got) >= self.IOU_THRESHOLD
                           for got in top))
        return found / len(expected)

    def mean_recalls(self, pairs: List[Pair],
                     ks: Sequence[int]) -> Dict[int, float]:
        """Average recall@k over (retrieved, expected) pairs, per k."""
        if not pairs:
            return {k: 0.0 for k in ks}
        return {k: sum(self.recall_at_k(r, e, k) for r, e in pairs)
                / len(pairs) for k in ks}

    def evaluate(self, search: StudentSearchResults, dataset: RagDataset,
                 ks: Sequence[int] = settings.recall_ks,
                 max_length: int = settings.max_context_length
                 ) -> EvaluationReport:
        """Recall@k of a whole search results file.

        Raises:
            ValueError: If the dataset has no answered question.
        """
        truth = {q.question_id: q.sources for q in dataset.rag_questions
                 if isinstance(q, AnsweredQuestion)}
        if not truth:
            raise ValueError("the dataset contains no answered questions")
        retrieved = {r.question_id: r.retrieved_sources
                     for r in search.search_results}
        too_long = sum(1 for r in search.search_results
                       for s in r.retrieved_sources if s.length > max_length)
        pairs = [(retrieved.get(qid, []), sources)
                 for qid, sources in truth.items()]
        return EvaluationReport(
            n_questions=len(truth),
            missing_results=sum(1 for qid in truth if qid not in retrieved),
            too_long_sources=too_long, max_context_length=max_length,
            recalls=self.mean_recalls(pairs, ks))


evaluator = Evaluator()
