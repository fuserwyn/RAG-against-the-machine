"""What every retriever offers: ``search(query, k)``."""

from typing import Any, Dict, List, Protocol, Tuple

from ..schemas.source import MinimalSource


Scored = Tuple[MinimalSource, float, Dict[str, Any]]


class SearchBackend(Protocol):
    """What callers need from a retriever, whatever the backend."""

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """Return the top-k sources for a query."""
        ...


def is_blank(query: str, k: int) -> bool:
    """Whether a search can only return nothing."""
    return k <= 0 or not query.strip()
