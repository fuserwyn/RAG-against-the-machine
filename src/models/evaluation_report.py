"""Result of a self-evaluation run (``evaluate`` command)."""

from typing import Dict

from pydantic import BaseModel


class EvaluationReport(BaseModel):
    """Recall@k of a search results file against a ground truth."""

    n_questions: int
    missing_results: int
    too_long_sources: int
    max_context_length: int
    recalls: Dict[int, float]

    def __str__(self) -> str:
        """The two lines printed by the CLI."""
        head = (f"Questions: {self.n_questions} (missing results: "
                f"{self.missing_results}, sources over "
                f"{self.max_context_length} chars: {self.too_long_sources})")
        scores = "  ".join(f"Recall@{k}: {v:.3f}"
                           for k, v in self.recalls.items())
        return f"{head}\n{scores}"
