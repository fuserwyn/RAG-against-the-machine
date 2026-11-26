"""VI.4 - Data models: the pydantic models given by the subject."""

import uuid
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from src.schemas import (AnsweredQuestion, MinimalAnswer, MinimalSearchResults,
                         MinimalSource, RagDataset, StudentSearchResults,
                         StudentSearchResultsAndAnswer, UnansweredQuestion)

from .helpers import DATASETS_DIR, requires_corpus

SOURCE = MinimalSource(file_path="data/raw/vllm-0.10.1/docs/a.md",
                       first_character_index=0, last_character_index=10)


def field_types(model: Any) -> Dict[str, str]:
    """``{field name: annotation}`` of a pydantic model."""
    return {name: str(info.annotation)
            for name, info in model.model_fields.items()}


def test_minimal_source_fields() -> None:
    """MinimalSource: file_path, first/last_character_index."""
    assert field_types(MinimalSource) == {
        "file_path": "<class 'str'>",
        "first_character_index": "<class 'int'>",
        "last_character_index": "<class 'int'>"}


def test_unanswered_question_gets_a_uuid4_by_default() -> None:
    """question_id defaults to a fresh uuid4 string."""
    first = UnansweredQuestion(question="a?")
    second = UnansweredQuestion(question="a?")
    assert uuid.UUID(first.question_id).version == 4
    assert first.question_id != second.question_id


def test_answered_question_extends_unanswered_question() -> None:
    """AnsweredQuestion adds sources and answer."""
    assert issubclass(AnsweredQuestion, UnansweredQuestion)
    question = AnsweredQuestion(question="q", sources=[SOURCE], answer="a")
    assert question.sources[0] == SOURCE
    assert set(AnsweredQuestion.model_fields) == {
        "question_id", "question", "sources", "answer"}


def test_rag_dataset_accepts_answered_and_unanswered_questions() -> None:
    """RagDataset holds a mix and keeps each question's real type."""
    dataset = RagDataset.model_validate({"rag_questions": [
        {"question_id": "a", "question": "q1",
         "sources": [SOURCE.model_dump()], "answer": "x"},
        {"question_id": "u", "question": "q2"}]})
    answered, unanswered = dataset.rag_questions
    assert isinstance(answered, AnsweredQuestion)
    assert type(unanswered) is UnansweredQuestion


def test_search_result_models() -> None:
    """MinimalSearchResults / MinimalAnswer have the subject's fields."""
    assert set(MinimalSearchResults.model_fields) == {
        "question_id", "question", "retrieved_sources"}
    assert issubclass(MinimalAnswer, MinimalSearchResults)
    assert set(MinimalAnswer.model_fields) == {
        "question_id", "question", "retrieved_sources", "answer"}


def test_student_result_models() -> None:
    """StudentSearchResults(AndAnswer): search_results + k."""
    results = StudentSearchResults(search_results=[MinimalSearchResults(
        question_id="q", question="?", retrieved_sources=[SOURCE])], k=5)
    answers = StudentSearchResultsAndAnswer(search_results=[MinimalAnswer(
        question_id="q", question="?", retrieved_sources=[], answer="a")],
        k=5)
    assert results.k == answers.k == 5
    assert set(StudentSearchResults.model_fields) == {"search_results", "k"}
    assert set(StudentSearchResultsAndAnswer.model_fields) == {
        "search_results", "k"}


@pytest.mark.parametrize("bad", [
    {"file_path": "x", "first_character_index": "zero",
     "last_character_index": 1},
    {"file_path": "x", "first_character_index": 0},
    {"first_character_index": 0, "last_character_index": 1},
])
def test_invalid_sources_are_rejected(bad: Dict[str, Any]) -> None:
    """Wrong types or missing fields raise a ValidationError."""
    with pytest.raises(ValidationError):
        MinimalSource.model_validate(bad)


@pytest.mark.parametrize("payload", [
    {"search_results": "nope", "k": 1},
    {"search_results": []},
    {"search_results": [{"question": "?", "retrieved_sources": []}],
     "k": 1},
])
def test_invalid_search_results_are_rejected(payload: Dict[str, Any]
                                             ) -> None:
    """Malformed search results never validate."""
    with pytest.raises(ValidationError):
        StudentSearchResults.model_validate(payload)


def test_models_roundtrip_through_json() -> None:
    """Dumped JSON validates back to an equal model."""
    original = StudentSearchResultsAndAnswer(search_results=[MinimalAnswer(
        question_id="q", question="?", retrieved_sources=[SOURCE],
        answer="a")], k=3)
    again = StudentSearchResultsAndAnswer.model_validate_json(
        original.model_dump_json())
    assert again == original


@requires_corpus
@pytest.mark.parametrize("scope,cls", [("AnsweredQuestions",
                                        AnsweredQuestion),
                                       ("UnansweredQuestions",
                                        UnansweredQuestion)])
@pytest.mark.parametrize("name", ["dataset_code_public.json",
                                  "dataset_docs_public.json"])
def test_public_datasets_validate(scope: str, cls: type, name: str) -> None:
    """The provided datasets parse into the right question type."""
    dataset = RagDataset.model_validate_json(
        (DATASETS_DIR / scope / name).read_text())
    questions: List[Any] = list(dataset.rag_questions)
    assert questions and all(type(q) is cls for q in questions)
