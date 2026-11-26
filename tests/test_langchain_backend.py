"""The alternative LangChain backend (same interface as the native one)."""

import json
from typing import Callable

import pytest

from src.repositories.corpus_repository import corpus_repository
from src.extra.langchain.chunker import langchain_chunker

from .helpers import MD_LONG, PY_SCHEDULER, CLIResult, MiniCorpus


@pytest.mark.parametrize("language,text", [("python", PY_SCHEDULER),
                                           ("markdown", MD_LONG),
                                           ("text", MD_LONG)])
@pytest.mark.parametrize("max_size", [200, 2000])
def test_langchain_spans_are_exact_and_bounded(language: str, text: str,
                                               max_size: int) -> None:
    """Recovered spans point at the exact text and fit max_size."""
    splitter = langchain_chunker.make_splitter(language, max_size)
    spans = langchain_chunker.split_spans(splitter, text, max_size)
    assert spans
    for start, end in spans:
        assert 0 <= start < end <= len(text)
        assert end - start <= max_size


def test_language_dispatch() -> None:
    """Python and Markdown get language-aware separators."""
    assert langchain_chunker.language_of("a/b.py") == "python"
    assert langchain_chunker.language_of("a/b.md") == "markdown"
    assert langchain_chunker.language_of("a/b.txt") == "text"


def test_langchain_backend_end_to_end(mini_corpus: MiniCorpus,
                                      cli: Callable[..., CLIResult]) -> None:
    """``--backend langchain`` indexes and searches like the native one."""
    built = cli("--backend", "langchain", "index", "--max_chunk_size",
                "500")
    assert built.code == 0, built.err
    found = cli("--backend", "langchain", "search",
                "load a LoRA adapter at runtime", "--k", "3")
    assert found.code == 0, found.err
    sources = json.loads(found.out)["retrieved_sources"]
    assert sources[0]["file_path"] == "data/raw/proj/docs/features/lora.md"
    for source in sources:
        text = corpus_repository.read_text(source["file_path"])
        assert 0 < source["last_character_index"] - \
            source["first_character_index"] <= 500
        assert source["last_character_index"] <= len(text)
