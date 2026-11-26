"""VI.5 - Output: JSON files that follow the pydantic models."""

import json
from pathlib import Path
from typing import Any, Callable, Dict

from src.models.index_build import IndexReport
from src.schemas import StudentSearchResults, StudentSearchResultsAndAnswer

from .helpers import CLIResult, GenerateCalls, MiniCorpus, fake_answer

QUESTIONS = {"rag_questions": [
    {"question_id": "q1", "question": "How do I load a LoRA adapter?"},
    {"question_id": "q2", "question": "What does the scheduler budget?"}]}


def write_dataset(corpus: MiniCorpus, name: str = "dataset.json") -> Path:
    """Write a two-question dataset and return its path."""
    path = corpus.root / name
    path.write_text(json.dumps(QUESTIONS))
    return path


def run_search(corpus: MiniCorpus, cli: Callable[..., CLIResult],
               k: str = "4") -> Dict[str, Any]:
    """Run search_dataset and return the parsed output file."""
    dataset = write_dataset(corpus)
    result = cli("search_dataset", "--dataset_path", str(dataset), "--k", k,
                 "--save_directory", "out/search")
    assert result.code == 0, result.err
    assert result.out.strip() == \
        "Saved student_search_results to out/search/dataset.json"
    data: Dict[str, Any] = json.loads(
        (corpus.root / "out/search/dataset.json").read_text())
    return data


def test_search_output_follows_student_search_results(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        cli: Callable[..., CLIResult]) -> None:
    """search_dataset writes StudentSearchResults with k."""
    data = run_search(mini_corpus, cli, k="4")
    parsed = StudentSearchResults.model_validate(data)
    assert set(data) == {"search_results", "k"} and parsed.k == 4
    first = data["search_results"][0]
    assert set(first) == {"question_id", "question", "retrieved_sources"}
    assert first["question_id"] == "q1"
    assert first["question"] == QUESTIONS["rag_questions"][0]["question"]
    source = first["retrieved_sources"][0]
    assert set(source) == {"file_path", "first_character_index",
                           "last_character_index"}
    assert source["file_path"].startswith("data/raw/")
    assert all(len(r["retrieved_sources"]) <= 4
               for r in data["search_results"])


def test_answer_output_follows_student_search_results_and_answer(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        fake_generator: GenerateCalls,
        cli: Callable[..., CLIResult]) -> None:
    """answer_dataset writes StudentSearchResultsAndAnswer."""
    search = run_search(mini_corpus, cli, k="3")
    result = cli("answer_dataset", "--student_search_results_path",
                 "out/search/dataset.json", "--save_directory", "out/answer")
    assert result.code == 0, result.err
    assert "Loaded 2 questions" in result.err
    assert "Processed 2 of 2 questions" in result.err
    data = json.loads((mini_corpus.root / "out/answer/dataset.json")
                      .read_text())
    parsed = StudentSearchResultsAndAnswer.model_validate(data)
    assert parsed.k == 3
    for got, before in zip(parsed.search_results, search["search_results"]):
        assert got.question_id == before["question_id"]
        assert [s.model_dump() for s in got.retrieved_sources] == \
            before["retrieved_sources"]
        assert got.answer == fake_answer(got.question,
                                         got.retrieved_sources)
    assert fake_generator.loads == 1


def test_output_file_keeps_the_dataset_name(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        cli: Callable[..., CLIResult]) -> None:
    """The output is ``<save_directory>/<dataset file name>``."""
    dataset = write_dataset(mini_corpus, "dataset_docs_public.json")
    result = cli("search_dataset", "--dataset_path", str(dataset),
                 "--save_directory", "deep/nested/dir")
    assert result.code == 0, result.err
    assert (mini_corpus.root / "deep/nested/dir/dataset_docs_public.json"
            ).is_file()
