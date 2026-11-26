"""LangChain chunking: ``RecursiveCharacterTextSplitter`` per language.

Python files use the ``Language.PYTHON`` separators (class / def first),
Markdown files the ``Language.MARKDOWN`` ones (headings first), other
text a generic paragraph / line / word cascade.
"""

from typing import Any, Dict, List

from ...models.chunk import Span
from ...services.span_splitter import SpanSplitter, span_splitter


class LangChainChunker:
    """Wraps the LangChain splitters and recovers exact spans."""

    languages = ("python", "markdown", "text")

    def __init__(self, splitter: SpanSplitter) -> None:
        """Use ``splitter`` to re-cut pieces over the limit."""
        self.splitter = splitter

    def language_of(self, path: str) -> str:
        """Pick the splitter from the file extension."""
        if path.endswith(".py"):
            return "python"
        if path.endswith((".md", ".rst")):
            return "markdown"
        return "text"

    def make_splitter(self, language: str, max_chunk_size: int) -> Any:
        """Build a LangChain splitter for ``language``."""
        from langchain_text_splitters import (Language,
                                              RecursiveCharacterTextSplitter)

        if language == "text":
            return RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size, chunk_overlap=0,
                add_start_index=True)
        lang = Language.PYTHON if language == "python" else Language.MARKDOWN
        return RecursiveCharacterTextSplitter.from_language(
            lang, chunk_size=max_chunk_size, chunk_overlap=0,
            add_start_index=True)

    def make_splitters(self, max_chunk_size: int) -> Dict[str, Any]:
        """One splitter per language."""
        return {name: self.make_splitter(name, max_chunk_size)
                for name in self.languages}

    def split_spans(self, splitter: Any, text: str,
                    max_chunk_size: int) -> List[Span]:
        """Run a splitter and recover exact ``[start, end)`` spans.

        LangChain only reports ``start_index``; the end is ``start + len``.
        A piece the splitter could not bring under the limit is re-cut on
        line boundaries, so no span exceeds ``max_chunk_size``.
        """
        spans: List[Span] = []
        for doc in splitter.create_documents([text]):
            start = int(doc.metadata.get("start_index", -1))
            if start < 0:
                continue
            end = start + len(doc.page_content)
            if end - start > max_chunk_size:
                spans.extend(self.splitter.split_by_lines(text, start, end,
                                                          max_chunk_size))
            else:
                spans.append((start, end))
        return spans


langchain_chunker = LangChainChunker(span_splitter)
