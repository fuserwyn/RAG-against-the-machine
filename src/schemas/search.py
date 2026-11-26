"""Search results and answers: the JSON files the CLI writes."""

from typing import List

from pydantic import BaseModel

from .source import MinimalSource


class MinimalSearchResults(BaseModel):
    """The ranked sources retrieved for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results for one question plus the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a whole dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results and answers for a whole dataset."""

    search_results: List[MinimalAnswer]
    k: int
