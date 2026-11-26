"""The LangChain index: one pickled ``BM25Retriever``."""

import os
import pickle
from typing import Any


class LangChainRepository:
    """Saves and loads ``<index_dir>/bm25_retriever.pkl``."""

    file_name = "bm25_retriever.pkl"

    def path(self, index_dir: str) -> str:
        """Location of the pickle."""
        return os.path.join(index_dir, self.file_name)

    def save(self, index_dir: str, retriever: Any) -> None:
        """Pickle the retriever (documents, metadata and BM25Okapi)."""
        os.makedirs(index_dir, exist_ok=True)
        with open(self.path(index_dir), "wb") as handle:
            pickle.dump(retriever, handle)

    def load(self, index_dir: str) -> Any:
        """Unpickle the retriever.

        Raises:
            FileNotFoundError: If the index has not been built yet.
        """
        if not os.path.isfile(self.path(index_dir)):
            raise FileNotFoundError(
                f"no langchain index found in {index_dir!r}: run "
                "'index --backend langchain' first")
        with open(self.path(index_dir), "rb") as handle:
            return pickle.load(handle)


langchain_repository = LangChainRepository()
