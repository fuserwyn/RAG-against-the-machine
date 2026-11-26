"""Pytest fixtures shared by every test module."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List

import numpy as np
import pytest

from src.models.index_build import IndexReport
from src.schemas import MinimalSource
from src.bonus.semantic.embedder import embedder_provider
from src.services.generator import generator_provider
from src.services.indexer import native_indexer

from .helpers import (CORPUS_DIR, PROJECT_ROOT, CLIResult, EmbedCalls,
                      GenerateCalls, MiniCorpus, fake_answer, hash_vectors,
                      run_cli, working_directory, write_corpus)


@pytest.fixture
def mini_corpus(tmp_path: Path,
                monkeypatch: pytest.MonkeyPatch) -> MiniCorpus:
    """A small corpus under ``tmp_path/data/raw``; cwd is ``tmp_path``."""
    monkeypatch.chdir(tmp_path)
    return write_corpus(tmp_path)


@pytest.fixture
def mini_index(mini_corpus: MiniCorpus) -> IndexReport:
    """The native index of :func:`mini_corpus` (default chunk size)."""
    return native_indexer.build(mini_corpus.raw_dir, mini_corpus.native_dir,
                                2000)


@pytest.fixture
def cli(capsys: pytest.CaptureFixture[str]) -> Callable[..., CLIResult]:
    """Run the CLI in-process: ``cli("search", "query", "--k", "3")``."""
    def run(*args: str) -> CLIResult:
        return run_cli(capsys, *args)
    return run


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> EmbedCalls:
    """Replace the MiniLM encoder by a fast hashing bag-of-words one."""
    calls = EmbedCalls()

    class FakeEmbedder:
        """Deterministic stand-in for :class:`Embedder`."""

        def __init__(self, model_name: str = "fake") -> None:
            """Count model loads."""
            calls.loads += 1

        def encode(self, texts: List[str], batch_size: int = 64,
                   progress: bool = False) -> np.ndarray:
            """Hash texts into normalized vectors."""
            calls.texts.extend(texts)
            return hash_vectors(texts)

    monkeypatch.setattr(embedder_provider, "factory", FakeEmbedder)
    return calls


@pytest.fixture
def fake_generator(monkeypatch: pytest.MonkeyPatch) -> GenerateCalls:
    """Replace Qwen by a generator that answers instantly."""
    calls = GenerateCalls()

    class FakeGenerator:
        """Deterministic stand-in for :class:`AnswerGenerator`."""

        def __init__(self, model_name: str = "fake") -> None:
            """Count model loads."""
            calls.loads += 1
            calls.models.append(model_name)

        def answer(self, question: str,
                   sources: List[MinimalSource]) -> str:
            """Return a canned answer."""
            calls.questions.append(question)
            return fake_answer(question, sources)

    monkeypatch.setattr(generator_provider, "factory", FakeGenerator)
    return calls


@dataclass
class CorpusIndex:
    """The native index of the real corpus, built once per session."""

    root: Path
    build_seconds: float

    @property
    def native_dir(self) -> str:
        """Directory of the native index."""
        return str(self.root / "native")


@pytest.fixture(scope="session")
def corpus_index(tmp_path_factory: pytest.TempPathFactory) -> CorpusIndex:
    """Index ``data/raw`` (the vLLM corpus) into a temporary directory."""
    if not CORPUS_DIR.is_dir():
        pytest.skip("the vLLM corpus is not in data/raw")
    root = tmp_path_factory.mktemp("corpus_index")
    with working_directory(PROJECT_ROOT):
        start = time.perf_counter()
        native_indexer.build("data/raw", str(root / "native"), 2000)
        elapsed = time.perf_counter() - start
    return CorpusIndex(root=root, build_seconds=elapsed)


@pytest.fixture
def in_project_root(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Run a test from the project root (corpus paths are relative)."""
    monkeypatch.chdir(PROJECT_ROOT)
    yield PROJECT_ROOT
