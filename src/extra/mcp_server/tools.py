"""What the MCP tools do: search the vLLM corpus, read a span of a file.

The tools reuse the RAG service: the client model decides
when to search and with which words, the BM25 index finds the places,
and the text of every hit is sent back, because the index itself only
stores addresses (file + character span).
"""

import os
from typing import List, Optional

from pydantic import BaseModel

from ...config import settings
from ...errors import RagError
from ...repositories.corpus_repository import (CorpusRepository,
                                               corpus_repository)
from ...services.rag_service import RagService


class SearchHit(BaseModel):
    """One retrieved chunk, with its text."""

    rank: int
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str


class SourceText(BaseModel):
    """A span of one corpus file."""

    file_path: str
    start: int
    end: int
    file_length: int
    text: str


class RagTools:
    """The logic behind the MCP tools, independent of the protocol."""

    max_k = 20
    max_read_chars = 6000

    def __init__(self, index_dir: str = settings.index_dir,
                 raw_dir: str = settings.raw_dir,
                 corpus: CorpusRepository = corpus_repository) -> None:
        """Search ``index_dir`` (native BM25) and read files of ``raw_dir``.

        The index is loaded on the first search, not at start-up, so the
        client sees the server immediately.
        """
        self.index_dir = index_dir
        self.raw_dir = raw_dir
        self.corpus = corpus
        self._service: Optional[RagService] = None

    def service(self) -> RagService:
        """The RAG service, created on first use."""
        if self._service is None:
            self._service = RagService(self.index_dir)
        return self._service

    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        """The ``k`` best chunks for ``query``, best first, with text.

        Raises:
            RagError: Empty query, ``k`` out of range, missing index.
        """
        if isinstance(k, bool) or not isinstance(k, int) \
                or not 1 <= k <= self.max_k:
            raise RagError(f"k must be an integer between 1 and "
                           f"{self.max_k}")
        found = self.service().search(query, k).retrieved_sources
        return [SearchHit(rank=rank, file_path=source.file_path,
                          first_character_index=source.first_character_index,
                          last_character_index=source.last_character_index,
                          text=self.corpus.read_span(source))
                for rank, source in enumerate(found, start=1)]

    def read_source(self, file_path: str, start: int = 0,
                    end: Optional[int] = None) -> SourceText:
        """Characters ``[start, end)`` of a corpus file.

        Raises:
            RagError: Path outside the corpus, missing file, bad range.
        """
        path = self._inside_corpus(file_path)
        try:
            text = self.corpus.read_text(path)
        except OSError as err:
            raise RagError(f"cannot read {file_path!r}: {err}") from err
        stop = len(text) if end is None else end
        if not 0 <= start <= stop:
            raise RagError("expected 0 <= start <= end")
        stop = min(stop, len(text), start + self.max_read_chars)
        return SourceText(file_path=file_path, start=start, end=stop,
                          file_length=len(text), text=text[start:stop])

    def _inside_corpus(self, file_path: str) -> str:
        """``file_path`` if it is a file under ``raw_dir``.

        Raises:
            RagError: Otherwise (``..``, absolute path elsewhere, ...).
        """
        root = os.path.realpath(self.raw_dir)
        real = os.path.realpath(file_path)
        if os.path.commonpath([root, real]) != root \
                or not os.path.isfile(real):
            raise RagError(f"{file_path!r} is not a file of the corpus "
                           f"({self.raw_dir})")
        return file_path
