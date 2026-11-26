"""VI.7 - End-to-end walkthrough and expected project layout."""

import json
from pathlib import Path
from typing import Callable

import pytest

from src.schemas import (RagDataset, StudentSearchResults,
                         StudentSearchResultsAndAnswer)

from .helpers import (CORPUS_DIR, DATASETS_DIR, PROJECT_ROOT, CLIResult,
                      GenerateCalls, MiniCorpus, requires_corpus)


@pytest.mark.parametrize("path", ["src/__main__.py", "src/__init__.py",
                                  "pyproject.toml", "uv.lock", "README.md",
                                  "Makefile"])
def test_repository_layout(path: str) -> None:
    """The files the reference scripts rely on are at the root."""
    assert (PROJECT_ROOT / path).is_file()


@requires_corpus
@pytest.mark.parametrize("path", [
    "data/raw", "data/datasets/UnansweredQuestions",
    "data/datasets/AnsweredQuestions"])
def test_data_layout(path: str) -> None:
    """Corpus and datasets are where the exam scripts expect them."""
    assert (PROJECT_ROOT / path).is_dir()
    assert CORPUS_DIR.is_dir()


def copy_dataset(corpus: MiniCorpus, scope: str) -> str:
    """Write a small dataset under data/datasets/<scope>/."""
    questions = [
        {"question_id": "q1", "question": "How to load a LoRA adapter at "
         "runtime?", "answer": "POST /v1/load_lora_adapter",
         "sources": [{"file_path": "data/raw/proj/docs/features/lora.md",
                      "first_character_index": 300,
                      "last_character_index": 420}]},
        {"question_id": "q2", "question": "Which knob does the "
         "zephyrquokka guide describe in paragraph 7?", "answer": "7",
         "sources": [{"file_path": "data/raw/proj/docs/zephyr.md",
                      "first_character_index": 900,
                      "last_character_index": 1100}]}]
    if scope == "UnansweredQuestions":
        questions = [{"question_id": q["question_id"],
                      "question": q["question"]} for q in questions]
    folder = corpus.root / "data/datasets" / scope
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "dataset_docs_public.json"
    path.write_text(json.dumps({"rag_questions": questions}))
    return str(path.relative_to(corpus.root))


def test_full_pipeline(mini_corpus: MiniCorpus,
                       fake_generator: GenerateCalls,
                       cli: Callable[..., CLIResult]) -> None:
    """index -> search_dataset -> evaluate -> answer_dataset."""
    unanswered = copy_dataset(mini_corpus, "UnansweredQuestions")
    answered = copy_dataset(mini_corpus, "AnsweredQuestions")

    step = cli("index", "--max_chunk_size", "2000")
    assert step.code == 0 and "Ingestion complete" in step.out
    assert (mini_corpus.root / "data/processed/native").is_dir()

    step = cli("search_dataset", "--dataset_path", unanswered, "--k", "10",
               "--save_directory",
               "data/output/search_results/UnansweredQuestions")
    results = ("data/output/search_results/UnansweredQuestions/"
               "dataset_docs_public.json")
    assert step.code == 0 and step.out.strip().endswith(results)
    search = StudentSearchResults.model_validate_json(
        (mini_corpus.root / results).read_text())
    assert search.k == 10 and len(search.search_results) == 2

    step = cli("evaluate", "--student_search_results_path", results,
               "--dataset_path", answered)
    assert step.code == 0, step.err
    assert "Recall@1: 1.000" in step.out and "Recall@10: 1.000" in step.out

    step = cli("answer_dataset", "--student_search_results_path", results,
               "--save_directory",
               "data/output/search_results_and_answer/UnansweredQuestions")
    assert step.code == 0, step.err
    answers = StudentSearchResultsAndAnswer.model_validate_json(
        (mini_corpus.root / "data/output/search_results_and_answer/"
         "UnansweredQuestions/dataset_docs_public.json").read_text())
    assert [a.question_id for a in answers.search_results] == ["q1", "q2"]
    assert all(a.answer for a in answers.search_results)


def test_scoped_save_directories_do_not_overwrite_each_other(
        mini_corpus: MiniCorpus, cli: Callable[..., CLIResult]) -> None:
    """Datasets sharing a file name are kept apart by scope."""
    cli("index")
    for scope in ("UnansweredQuestions", "AnsweredQuestions"):
        dataset = copy_dataset(mini_corpus, scope)
        step = cli("search_dataset", "--dataset_path", dataset,
                   "--save_directory", f"data/output/search_results/{scope}")
        assert step.code == 0, step.err
    for scope in ("UnansweredQuestions", "AnsweredQuestions"):
        assert (mini_corpus.root / "data/output/search_results" / scope /
                "dataset_docs_public.json").is_file()


@requires_corpus
def test_public_dataset_ids_survive_the_pipeline(
        tmp_path: Path, cli: Callable[..., CLIResult],
        in_project_root: object) -> None:
    """search_dataset on a real public dataset keeps every question id."""
    dataset = DATASETS_DIR / "UnansweredQuestions/dataset_docs_public.json"
    index_root = str(tmp_path / "index")
    assert cli("--index_dir", index_root, "index").code == 0
    step = cli("--index_dir", index_root, "search_dataset",
               "--dataset_path", str(dataset), "--k", "10",
               "--save_directory", str(tmp_path))
    assert step.code == 0, step.err
    expected = RagDataset.model_validate_json(dataset.read_text())
    got = StudentSearchResults.model_validate_json(
        (tmp_path / "dataset_docs_public.json").read_text())
    assert [r.question_id for r in got.search_results] == \
        [q.question_id for q in expected.rag_questions]
