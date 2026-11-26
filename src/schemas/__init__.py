"""Schemas: the pydantic models exchanged with the outside world.

These are the structures from the subject (datasets in, search results
and answers out). They are validated on every read and write.
"""

from .question import AnsweredQuestion, RagDataset, UnansweredQuestion
from .search import (MinimalAnswer, MinimalSearchResults, StudentSearchResults,
                     StudentSearchResultsAndAnswer)
from .source import MinimalSource, sources_adapter

__all__ = [
    "AnsweredQuestion", "MinimalAnswer", "MinimalSearchResults",
    "MinimalSource", "RagDataset", "StudentSearchResults",
    "StudentSearchResultsAndAnswer", "UnansweredQuestion", "sources_adapter",
]
