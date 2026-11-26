"""VI.2 - Retrieval: top-k sources, exact paths, single and batch search."""

import json
import os
import time
from typing import Any, Callable, Dict, List

import pytest

from src.models.index_build import IndexReport
from src.schemas import (MinimalSearchResults, MinimalSource, RagDataset,
                         StudentSearchResults)
from src.services.evaluator import evaluator
from src.services.ranker import ranker
from src.services.retriever import LexicalRetriever
from src.services.tokenizer import tokenizer
from src.services.rag_service import RagService
from src.views.cli import RagCLI

from .conftest import CorpusIndex
from .helpers import (DATASETS_DIR, PROJECT_ROOT, CLIResult, MiniCorpus,
                      requires_corpus)

LORA_MD = "data/raw/proj/docs/features/lora.md"


def search(corpus: MiniCorpus, query: str, k: int) -> List[MinimalSource]:
    """Search the mini index."""
    return LexicalRetriever(corpus.native_dir).search(query, k)


def test_relevant_file_is_ranked_first(mini_corpus: MiniCorpus,
                                       mini_index: IndexReport) -> None:
    """A question about runtime LoRA loading hits the LoRA doc first."""
    sources = search(mini_corpus, "How do I load a LoRA adapter at runtime?",
                     3)
    assert sources[0].file_path == LORA_MD


def test_verbatim_identifier_is_found(mini_corpus: MiniCorpus,
                                      mini_index: IndexReport) -> None:
    """Quoting an identifier finds the code that defines it."""
    sources = search(mini_corpus, "What is FusedMoEActivationFormat?", 1)
    assert sources[0].file_path == "data/raw/proj/vllm/core/scheduler.py"


def test_paraphrased_identifier_is_found(mini_corpus: MiniCorpus,
                                         mini_index: IndexReport) -> None:
    """Splitting identifiers lets plain words match ``snake_case``."""
    sources = search(mini_corpus, "maximum num seqs constant", 1)
    text = (mini_corpus.root / sources[0].file_path).read_text()
    assert "MAX_NUM_SEQS" in text[sources[0].first_character_index:
                                  sources[0].last_character_index]


@pytest.mark.parametrize("k", [1, 2, 5])
def test_at_most_k_ranked_sources(mini_corpus: MiniCorpus,
                                  mini_index: IndexReport, k: int) -> None:
    """At most k sources are returned, best first."""
    retriever = LexicalRetriever(mini_corpus.native_dir)
    query = "scheduler budget method"
    sources = retriever.search(query, k)
    assert 1 <= len(sources) <= k
    scores = retriever.bm25.score(tokenizer.tokenize(query))
    ranked = [scores[i] for i in retriever.ranking(query, k)]
    assert ranked == sorted(ranked, reverse=True)


def test_k_larger_than_the_index(mini_corpus: MiniCorpus,
                                 mini_index: IndexReport) -> None:
    """Asking for more sources than chunks returns every match."""
    sources = search(mini_corpus, "zephyrquokka", 10_000)
    assert 0 < len(sources) <= mini_index.n_chunks


@pytest.mark.parametrize("query", ["", "   ", "xqzzv blorptang"])
def test_empty_or_nonsensical_query_returns_nothing(
        mini_corpus: MiniCorpus, mini_index: IndexReport, query: str) -> None:
    """No match is an empty list, never an exception."""
    assert search(mini_corpus, query, 5) == []


def test_k_zero_returns_nothing(mini_corpus: MiniCorpus,
                                mini_index: IndexReport) -> None:
    """The retriever itself tolerates k = 0."""
    assert search(mini_corpus, "lora", 0) == []


def test_sources_are_exact_corpus_paths_and_spans(
        mini_corpus: MiniCorpus, mini_index: IndexReport) -> None:
    """file_path is the corpus path verbatim; spans are <= 2000 chars."""
    for source in search(mini_corpus, "scheduler lora kernel notes", 50):
        assert source.file_path in mini_corpus.files
        assert source.file_path.startswith("data/raw/")
        size = source.last_character_index - source.first_character_index
        assert 0 < size <= 2000
        assert source.last_character_index <= \
            len(mini_corpus.files[source.file_path])


def test_top_indices_orders_and_filters() -> None:
    """top_indices returns the best scores first, optionally > 0 only."""
    import numpy as np

    scores = np.array([0.0, 3.0, 1.0, 2.0, 0.0])
    assert ranker.top_indices(scores, 3) == [1, 3, 2]
    assert ranker.top_indices(scores, 10, positive_only=True) == [1, 3, 2]
    assert ranker.top_indices(scores, 0) == []


def test_single_query_command(mini_corpus: MiniCorpus,
                              mini_index: IndexReport,
                              cli: Callable[..., CLIResult]) -> None:
    """``search <query> --k`` prints MinimalSearchResults JSON."""
    result = cli("search", "load lora adapter", "--k", "2")
    assert result.code == 0, result.err
    parsed = MinimalSearchResults.model_validate_json(result.out)
    assert parsed.question == "load lora adapter"
    assert 0 < len(parsed.retrieved_sources) <= 2


def test_batch_search_over_a_json_dataset(mini_corpus: MiniCorpus,
                                          mini_index: IndexReport,
                                          cli: Callable[..., CLIResult]
                                          ) -> None:
    """``search_dataset`` answers every question of a JSON dataset."""
    dataset = {"rag_questions": [
        {"question_id": "q1", "question": "load a LoRA adapter at runtime"},
        {"question_id": "q2", "question": "What is FusedMoEActivationFormat"},
        {"question": "zephyrquokka tuning knob"}]}
    path = mini_corpus.root / "questions.json"
    path.write_text(json.dumps(dataset))
    result = cli("search_dataset", "--dataset_path", str(path), "--k", "3",
                 "--save_directory", "out")
    assert result.code == 0, result.err
    output = StudentSearchResults.model_validate_json(
        (mini_corpus.root / "out" / "questions.json").read_text())
    assert [r.question_id for r in output.search_results][:2] == ["q1", "q2"]
    assert all(r.retrieved_sources for r in output.search_results)
    assert output.search_results[0].retrieved_sources[0].file_path == \
        LORA_MD


# ------------------------------------------------------ reference datasets

def recall_at_5(corpus_index: CorpusIndex, name: str) -> float:
    """Recall@5 of the native index on a public dataset."""
    service = RagService(index_dir=str(corpus_index.root), cache=False)
    retriever = service.retriever()
    answered = RagDataset.model_validate_json(
        (DATASETS_DIR / "AnsweredQuestions" / name).read_text())
    pairs = [(retriever.search(q.question, 5), getattr(q, "sources"))
             for q in answered.rag_questions]
    return evaluator.mean_recalls(pairs, (5,))[5]


@requires_corpus
def test_docs_recall_at_5_is_at_least_80_percent(
        corpus_index: CorpusIndex, in_project_root: object) -> None:
    """Docs questions: recall@5 >= 0.80."""
    assert recall_at_5(corpus_index, "dataset_docs_public.json") >= 0.80


@requires_corpus
def test_code_recall_at_5_is_at_least_50_percent(
        corpus_index: CorpusIndex, in_project_root: object) -> None:
    """Code questions: recall@5 >= 0.50."""
    assert recall_at_5(corpus_index, "dataset_code_public.json") >= 0.50


@requires_corpus
def test_200_questions_are_searched_in_under_90_seconds(
        corpus_index: CorpusIndex, in_project_root: object,
        tmp_path: os.PathLike[str]) -> None:
    """Retrieval throughput: 200 questions in at most 90 s (cold start)."""
    questions: List[Dict[str, Any]] = []
    for name in ("dataset_code_public.json", "dataset_docs_public.json"):
        data = RagDataset.model_validate_json(
            (DATASETS_DIR / "UnansweredQuestions" / name).read_text())
        questions.extend(q.model_dump() for q in data.rag_questions)
    while len(questions) < 200:
        questions.append(dict(questions[len(questions) % 50],
                              question_id=f"extra-{len(questions)}"))
    path = os.path.join(tmp_path, "two_hundred.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"rag_questions": questions[:200]}, handle)
    start = time.perf_counter()
    service = RagCLI(index_dir=str(corpus_index.root), cache=False)
    service.search_dataset(path, 10, os.path.join(tmp_path, "out"))
    assert time.perf_counter() - start < 90


@requires_corpus
def test_retrieved_paths_exist_in_the_corpus(
        corpus_index: CorpusIndex, in_project_root: object) -> None:
    """Every retrieved path is a real file of data/raw/vllm-0.10.1/."""
    retriever = LexicalRetriever(corpus_index.native_dir)
    for source in retriever.search("How to configure the OpenAI server?",
                                   10):
        assert source.file_path.startswith("data/raw/vllm-0.10.1/")
        assert (PROJECT_ROOT / source.file_path).is_file()
