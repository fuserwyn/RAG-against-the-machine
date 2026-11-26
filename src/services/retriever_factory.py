"""Builds the retriever of a (backend, mode) pair, with clear errors."""

from ..bonus.hybrid.hybrid_retriever import HybridRetriever
from ..bonus.semantic.semantic_retriever import SemanticRetriever
from ..errors import RagError
from ..extra.langchain.retriever import LangChainRetriever
from ..extra.qdrant.retriever import QdrantRetriever
from .retriever import LexicalRetriever
from .search_backend import SearchBackend


class RetrieverFactory:
    """Maps the CLI options to a retriever class."""

    def create(self, backend: str, mode: str,
               index_dir: str) -> SearchBackend:
        """Load the index of ``backend`` / ``mode`` from ``index_dir``.

        Raises:
            RagError: Missing or corrupted index, missing dependency.
        """
        try:
            return self._build(backend, mode, index_dir)
        except (FileNotFoundError, RuntimeError) as err:
            raise RagError(str(err)) from err
        except ImportError as err:
            raise RagError(f"missing dependency ({err}); run "
                           "'uv sync'") from err
        except Exception as err:  # noqa: BLE001 - corrupted index
            raise RagError(f"corrupted index in {index_dir}: {err}") from err

    @staticmethod
    def _build(backend: str, mode: str, index_dir: str) -> SearchBackend:
        """The retriever itself (may raise any loading error)."""
        if backend == "langchain":
            return LangChainRetriever(index_dir)
        if backend == "qdrant":
            return QdrantRetriever(index_dir)
        if mode == "semantic":
            return SemanticRetriever(index_dir)
        if mode == "hybrid":
            return HybridRetriever(index_dir)
        return LexicalRetriever(index_dir)


retriever_factory = RetrieverFactory()
