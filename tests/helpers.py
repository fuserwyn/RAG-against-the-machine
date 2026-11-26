"""Shared test helpers: a tiny corpus, fake models and a CLI runner."""

import os
import sys
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, TypeVar
from unittest.mock import patch

import numpy as np
import pytest

from src.schemas import MinimalSource
from src.services.tokenizer import tokenizer
from src.views.cli import main

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "raw" / "vllm-0.10.1"
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
QWEN = "Qwen/Qwen3-0.6B"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"

F = TypeVar("F", bound=Callable[..., Any])


def requires_corpus(func: F) -> F:
    """Mark a test that needs the real vLLM corpus and datasets."""
    missing = not CORPUS_DIR.is_dir() or not DATASETS_DIR.is_dir()
    skip = pytest.mark.skipif(missing, reason="data/raw or data/datasets "
                              "is missing")
    marked: F = pytest.mark.corpus(skip(func))
    return marked


def model_available(repo_id: str) -> bool:
    """Whether a Hugging Face model is already in the local cache."""
    if os.environ.get("RAG_SKIP_MODEL_TESTS"):
        return False
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)


def requires_model(repo_id: str) -> Callable[[F], F]:
    """Mark a test that loads a real (cached) Hugging Face model."""
    def decorate(func: F) -> F:
        skip = pytest.mark.skipif(not model_available(repo_id),
                                  reason=f"{repo_id} is not downloaded")
        marked: F = pytest.mark.model(skip(func))
        return marked
    return decorate


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    """Temporarily change the current directory."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


# --------------------------------------------------------------------------
# A tiny corpus with every kind of file the indexer must handle.
# --------------------------------------------------------------------------

PY_SCHEDULER = '''"""Scheduler for the paged attention engine."""

import os

MAX_NUM_SEQS = 256


def helper(value: int) -> int:
    """Return value plus one."""
    return value + 1


# Comment that belongs to the decorated function.
@decorator_one
@decorator_two(arg=1)
def decorated_function():
    """A decorated function."""
    return 42


class FusedMoEActivationFormat:
    """Activation layout of the fused mixture-of-experts kernel."""

    STANDARD = 1
    BATCHED = 2


class TokenScheduler:
    """Schedules sequences with a preemption policy."""

''' + "".join(
    f'''    def method_{i:02d}(self, budget: int) -> int:
        """Method number {i} of the scheduler (budget planning)."""
        return budget * {i} + len(str(budget))

''' for i in range(30))

PY_BROKEN = "def broken(:\n    return 'this file does not parse'\n" * 3

MD_LORA = """# LoRA Adapters

LoRA (low-rank adaptation) adapters can be served by the engine.

## Serving adapters

Start the server with `--enable-lora` to serve adapters.

```bash
# this line is a shell comment, not a heading
vllm serve base-model --enable-lora
```

## Dynamic loading

To load a LoRA adapter at runtime, send a POST request to the
`/v1/load_lora_adapter` endpoint with the adapter name and path.
"""

MD_LONG = "# Zephyrquokka guide\n\n" + "".join(
    f"Paragraph {i} explains the zephyrquokka tuning knob number {i} "
    "in a reasonably long sentence that fills the section.\n\n"
    for i in range(60))

TXT_NOTES = "Plain text notes.\n\nThe quantization kernel uses int8.\n"

MD_CRLF = "# Windows file\r\n\r\nLine endings are CRLF here.\r\n\r\n" \
    "## Second heading\r\n\r\nSpeculative decoding with a draft model.\r\n"

CORPUS_FILES: Dict[str, str] = {
    "data/raw/proj/vllm/core/scheduler.py": PY_SCHEDULER,
    "data/raw/proj/vllm/broken.py": PY_BROKEN,
    "data/raw/proj/docs/features/lora.md": MD_LORA,
    "data/raw/proj/docs/zephyr.md": MD_LONG,
    "data/raw/proj/docs/windows.md": MD_CRLF,
    "data/raw/proj/notes.txt": TXT_NOTES,
    "data/raw/proj/empty.md": "",
    "data/raw/proj/kernel.cu": "__global__ void ignored_kernel() {}\n",
    "data/raw/proj/.git/ignored.py": "def ignored(): pass\n",
}


@dataclass
class MiniCorpus:
    """A corpus written under ``root/data/raw`` (``root`` is the cwd)."""

    root: Path
    files: Dict[str, str]
    raw_dir: str = "data/raw"
    index_root: str = "data/processed"

    @property
    def native_dir(self) -> str:
        """Directory of the native index."""
        return os.path.join(self.index_root, "native")

    def write(self, rel_path: str, content: str) -> None:
        """Create or overwrite one corpus file (bytes kept as given)."""
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        self.files[rel_path] = content


def write_corpus(root: Path) -> MiniCorpus:
    """Write :data:`CORPUS_FILES` under ``root``."""
    corpus = MiniCorpus(root=root, files={})
    for rel_path, content in CORPUS_FILES.items():
        corpus.write(rel_path, content)
    return corpus


# --------------------------------------------------------------------------
# Fake models: deterministic, fast, and they record how they were used.
# --------------------------------------------------------------------------

FAKE_DIM = 64


def hash_vectors(texts: List[str]) -> np.ndarray:
    """Bag-of-words vectors hashed into FAKE_DIM buckets, L2-normalized."""
    vectors = np.zeros((len(texts), FAKE_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in tokenizer.tokenize(text):
            vectors[row, zlib.crc32(token.encode()) % FAKE_DIM] += 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    result: np.ndarray = vectors / np.maximum(norms, 1e-9)
    return result


@dataclass
class EmbedCalls:
    """What the fake embedder was asked to do."""

    loads: int = 0
    texts: List[str] = field(default_factory=list)


@dataclass
class GenerateCalls:
    """What the fake generator was asked to do."""

    loads: int = 0
    models: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)


def fake_answer(question: str, sources: List[MinimalSource]) -> str:
    """The deterministic answer produced by the fake generator."""
    return f"Answer to '{question}' from {len(sources)} source(s)."


# --------------------------------------------------------------------------
# Running the CLI in-process.
# --------------------------------------------------------------------------

@dataclass
class CLIResult:
    """Exit code and captured output of one CLI run."""

    code: int
    out: str
    err: str


def run_cli(capsys: pytest.CaptureFixture[str], *args: str) -> CLIResult:
    """Run ``python -m src <args>`` in-process, as Fire would."""
    capsys.readouterr()
    code = 0
    with patch.object(sys, "argv", ["src", *args]):
        try:
            main()
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    captured = capsys.readouterr()
    return CLIResult(code=code, out=captured.out, err=captured.err)
