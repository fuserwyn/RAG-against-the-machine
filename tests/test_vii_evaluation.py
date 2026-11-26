"""VII - Evaluation: recall@k with an IoU >= 0.05 overlap rule."""

import json
from typing import Callable, List

import pytest

from src.schemas import MinimalSource
from src.services.evaluator import evaluator

from .helpers import CLIResult, MiniCorpus


def src(start: int, end: int, path: str = "a.md") -> MinimalSource:
    """Shorthand for a source span."""
    return MinimalSource(file_path=path, first_character_index=start,
                         last_character_index=end)


def test_iou_threshold_is_005() -> None:
    """The overlap bar is an IoU of 0.05."""
    assert evaluator.IOU_THRESHOLD == 0.05


@pytest.mark.parametrize("a,b,expected", [
    (src(0, 100), src(0, 100), 1.0),
    (src(0, 100), src(50, 150), 50 / 150),
    (src(0, 100), src(100, 200), 0.0),
    (src(0, 100), src(200, 300), 0.0),
    (src(0, 100), src(10, 20), 0.1),
    (src(0, 100), src(0, 100, "b.md"), 0.0),
])
def test_iou(a: MinimalSource, b: MinimalSource, expected: float) -> None:
    """Intersection over union of two character spans."""
    assert evaluator.iou(a, b) == pytest.approx(expected)
    assert evaluator.iou(b, a) == pytest.approx(expected)


def test_a_different_file_never_counts() -> None:
    """Same span, other file: not found (file_path must be exact)."""
    expected = [src(0, 100, "data/raw/vllm-0.10.1/docs/a.md")]
    retrieved = [src(0, 100, "vllm-0.10.1/docs/a.md")]
    assert evaluator.recall_at_k(retrieved, expected, 10) == 0.0


def test_small_overlap_above_the_bar_counts() -> None:
    """A big chunk covering a small reference span still counts."""
    assert evaluator.recall_at_k([src(0, 2000)], [src(500, 600)], 1) == 1.0


def test_overlap_below_the_bar_does_not_count() -> None:
    """IoU under 0.05 (4 / 2000 here) is a miss."""
    assert evaluator.iou(src(0, 2000), src(1996, 2004)) < 0.05
    assert evaluator.recall_at_k([src(0, 2000)], [src(1996, 2004)], 1) == 0.0


def test_only_the_top_k_results_count() -> None:
    """A correct source at rank 3 counts for k = 3, not for k = 1."""
    retrieved = [src(900, 950), src(800, 850), src(0, 100)]
    assert evaluator.recall_at_k(retrieved, [src(0, 100)], 1) == 0.0
    assert evaluator.recall_at_k(retrieved, [src(0, 100)], 3) == 1.0


def test_recall_is_the_share_of_found_sources() -> None:
    """Two references, one found: recall 0.5."""
    expected = [src(0, 100), src(5000, 5100)]
    assert evaluator.recall_at_k([src(0, 100)], expected, 5) == 0.5


def test_mean_recalls_averages_over_questions() -> None:
    """Dataset recall@k is the mean of per-question recall@k."""
    pairs: List[tuple[List[MinimalSource], List[MinimalSource]]] = [
        ([src(0, 100)], [src(0, 100)]),
        ([src(500, 600)], [src(0, 100)]),
    ]
    assert evaluator.mean_recalls(pairs, (1, 5)) == {1: 0.5, 5: 0.5}
    assert evaluator.mean_recalls([], (1,)) == {1: 0.0}


def test_evaluate_command(mini_corpus: MiniCorpus,
                          cli: Callable[..., CLIResult]) -> None:
    """``evaluate`` prints recall@1/3/5/10 and flags oversized sources."""
    results = {"k": 10, "search_results": [
        {"question_id": "q1", "question": "?", "retrieved_sources": [
            src(5000, 5100).model_dump(), src(0, 2500).model_dump()]},
        {"question_id": "q2", "question": "?", "retrieved_sources": []}]}
    dataset = {"rag_questions": [
        {"question_id": "q1", "question": "?", "answer": "",
         "sources": [src(10, 300).model_dump()]},
        {"question_id": "q2", "question": "?", "answer": "",
         "sources": [src(0, 10).model_dump()]},
        {"question_id": "q3", "question": "?", "answer": "",
         "sources": [src(0, 10).model_dump()]}]}
    (mini_corpus.root / "r.json").write_text(json.dumps(results))
    (mini_corpus.root / "d.json").write_text(json.dumps(dataset))
    result = cli("evaluate", "--student_search_results_path", "r.json",
                 "--dataset_path", "d.json")
    assert result.code == 0, result.err
    assert "Questions: 3 (missing results: 1, sources over 2000 chars: 1)" \
        in result.out
    assert "Recall@1: 0.000  Recall@3: 0.333  Recall@5: 0.333  " \
        "Recall@10: 0.333" in result.out


def test_solution_never_uses_the_moulinette() -> None:
    """The official grader is never imported or called by the code."""
    from .helpers import PROJECT_ROOT

    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        assert "moulinette" not in path.read_text().lower(), path
