"""A caching front for any retriever (bonus 4).

The real retriever is only built on the first cache miss: a repeated
query is answered without loading the index, the encoder or the LLM,
which is what makes repeated queries and cold starts fast.
"""

import threading
from typing import Callable, List, Optional, Sequence

from pydantic import ValidationError

from ...schemas.source import MinimalSource, sources_adapter
from ...services.search_backend import SearchBackend
from .cache_repository import CacheRepository


class CachedSearch:
    """A retriever wrapper that loads the real retriever on first miss."""

    def __init__(self, factory: Callable[[], SearchBackend],
                 cache: Optional[CacheRepository],
                 scope: Sequence[str]) -> None:
        """Wrap the retriever built by ``factory``.

        Args:
            factory: Builds the real retriever (called at most once).
            cache: Where to store results; ``None`` disables caching.
            scope: Key prefix (backend, mode, index build id).
        """
        self._factory = factory
        self._cache = cache
        self._scope = list(scope)
        self._backend: Optional[SearchBackend] = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """Whether the real retriever has been loaded."""
        return self._backend is not None

    def load(self) -> SearchBackend:
        """Return the real retriever, building it on first use."""
        with self._lock:
            if self._backend is None:
                self._backend = self._factory()
            return self._backend

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """Cached top-k search."""
        if self._cache is None:
            return self._search(query, k)
        key = CacheRepository.make_key("search", *self._scope, query, k)
        hit = self._cache.get(key)
        if hit is not None:
            try:
                return sources_adapter.validate_json(hit)
            except ValidationError:
                pass
        result = self._search(query, k)
        self._cache.set(key, sources_adapter.dump_json(result).decode("utf-8"))
        return result

    def _search(self, query: str, k: int) -> List[MinimalSource]:
        """Run the real retriever (searches are serialized for safety)."""
        backend = self.load()
        with self._lock:
            return backend.search(query, k)
