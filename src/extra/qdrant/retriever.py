"""Qdrant retriever: nearest MiniLM vectors."""

from typing import Any, List

from ...bonus.semantic.embedder import embedder_provider
from ...repositories.meta_repository import meta_repository
from ...schemas.source import MinimalSource
from ...services.search_backend import Scored, is_blank
from .qdrant_repository import qdrant_repository


class QdrantRetriever:
    """Embed the question, then ask Qdrant for its nearest chunks."""

    def __init__(self, index_dir: str) -> None:
        """Open the local Qdrant storage and load the encoder.

        Raises:
            FileNotFoundError: If the index has not been built yet.
            RuntimeError: If another process already holds the storage.
        """
        self.meta = meta_repository.read(index_dir)
        if not self.meta:
            raise FileNotFoundError(
                f"no qdrant index found in {index_dir!r}: run "
                "'--backend qdrant index' first")
        self.client: Any = qdrant_repository.open(index_dir)
        self.embedder = embedder_provider.create(
            str(self.meta.get("model", embedder_provider.default_model)))

    def scored(self, query: str, k: int) -> List[Scored]:
        """Top-k sources with their cosine score and full payload."""
        if is_blank(query, k):
            return []
        vector = self.embedder.encode([query])[0]
        results: List[Scored] = []
        for hit in qdrant_repository.nearest(self.client, vector, k):
            payload = dict(hit.payload or {})
            source = MinimalSource(
                file_path=payload["file_path"],
                first_character_index=payload["first_character_index"],
                last_character_index=payload["last_character_index"])
            results.append((source, float(hit.score), payload))
        return results

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """Return the ``k`` nearest source spans for ``query``."""
        return [source for source, _, _ in self.scored(query, k)]
