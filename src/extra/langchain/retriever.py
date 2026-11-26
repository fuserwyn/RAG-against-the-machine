"""LangChain retriever: the pickled ``BM25Retriever``."""

from typing import Any, List

from ...schemas.source import MinimalSource
from ...services.search_backend import is_blank
from ...services.tokenizer import tokenizer
from .langchain_repository import langchain_repository


class LangChainRetriever:
    """The pickled LangChain ``BM25Retriever`` behind ``search(query, k)``."""

    def __init__(self, index_dir: str) -> None:
        """Load the retriever built by the LangChain indexer.

        Raises:
            FileNotFoundError: If the index has not been built yet.
        """
        self.retriever: Any = langchain_repository.load(index_dir)

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """Return the ``k`` best source spans for ``query``."""
        if is_blank(query, k) or not tokenizer.tokenize(query):
            return []
        self.retriever.k = k
        docs = self.retriever.invoke(query)
        return [MinimalSource(**doc.metadata) for doc in docs]
