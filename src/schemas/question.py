"""Question datasets: the input of ``search_dataset`` and ``evaluate``."""

import uuid
from typing import List

from pydantic import BaseModel, Field

from .source import MinimalSource


class UnansweredQuestion(BaseModel):
    """A question without an answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of (answered or unanswered) RAG questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]
