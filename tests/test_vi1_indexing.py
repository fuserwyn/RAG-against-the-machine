"""VI.1 - Indexing the codebase: chunking strategies, BM25, persistence."""

import math
import os
from typing import Callable, List

import numpy as np
import pytest

from src.models.bm25_index import BM25Index
from src.repositories.bm25_repository import bm25_repository
from src.models.chunk import Chunk
from src.models.index_build import IndexReport
from src.repositories.chunk_repository import chunk_repository
from src.repositories.corpus_repository import corpus_repository
from src.services.chunkers import (file_chunker, markdown_chunker,
                                   python_chunker, text_chunker)
from src.services.indexer import native_indexer
from src.services.span_splitter import span_splitter
from src.services.tokenizer import tokenizer

from .conftest import CorpusIndex
from .helpers import (MD_CRLF, MD_LONG, MD_LORA, PY_BROKEN, PY_SCHEDULER,
                      CLIResult, MiniCorpus, requires_corpus)

SIZES = [120, 500, 2000]
PY_PATH = "data/raw/proj/vllm/core/scheduler.py"


def spans_of(chunks: List[Chunk]) -> List[tuple[int, int]]:
    """The ``(start, end)`` span of every chunk."""
    return [(c.first_character_index, c.last_character_index)
            for c in chunks]


def assert_valid_chunks(chunks: List[Chunk], text: str,
                        max_size: int) -> None:
    """Chunks are ordered, disjoint, <= max_size and cover all content."""
    assert chunks
    previous_end = 0
    for start, end in spans_of(chunks):
        assert 0 <= start < end <= len(text)
        assert end - start <= max_size
        assert start >= previous_end
        assert text[previous_end:start].strip() == ""
        previous_end = end
    assert text[previous_end:].strip() == ""


# ----------------------------------------------------- Python code chunking

@pytest.mark.parametrize("max_size", SIZES)
def test_python_chunks_respect_the_size_and_cover_the_file(
        max_size: int) -> None:
    """No chunk exceeds max_chunk_size and no code is lost."""
    chunks = python_chunker.chunk(PY_SCHEDULER, PY_PATH, max_size,
                                  max_size // 2)
    assert_valid_chunks(chunks, PY_SCHEDULER, max_size)
    assert all(c.kind == "python" for c in chunks)


def test_python_chunks_start_at_definitions() -> None:
    """Boundaries follow the syntax tree: a definition is never cut."""
    chunks = python_chunker.chunk(PY_SCHEDULER, PY_PATH, 2000, 1)
    starts = [PY_SCHEDULER[c.first_character_index:].split("\n", 1)[0]
              for c in chunks]
    assert "def helper(value: int) -> int:" in starts
    assert "class FusedMoEActivationFormat:" in starts


def test_python_decorators_and_comments_stay_with_their_function() -> None:
    """A decorated function keeps its decorators and leading comment."""
    chunks = python_chunker.chunk(PY_SCHEDULER, PY_PATH, 2000, 1)
    body = next(PY_SCHEDULER[c.first_character_index:
                             c.last_character_index]
                for c in chunks if "def decorated_function" in
                PY_SCHEDULER[c.first_character_index:c.last_character_index])
    assert "@decorator_one" in body and "@decorator_two(arg=1)" in body
    assert "# Comment that belongs" in body


def test_big_classes_are_split_per_method_with_class_context() -> None:
    """A class over the limit is cut at its methods, named in context."""
    chunks = python_chunker.chunk(PY_SCHEDULER, PY_PATH, 500, 250)
    methods = [c for c in chunks if c.context == "TokenScheduler"]
    assert len(methods) > 1
    for chunk in methods:
        text = PY_SCHEDULER[chunk.first_character_index:
                            chunk.last_character_index]
        assert text.lstrip().startswith(("def method_", "class "))


def test_unparsable_python_falls_back_to_text_chunking() -> None:
    """A file with a syntax error is still indexed (as text)."""
    chunks = python_chunker.chunk(PY_BROKEN, "data/raw/x/broken.py",
                                  2000, 1000)
    assert_valid_chunks(chunks, PY_BROKEN, 2000)
    assert {c.kind for c in chunks} == {"text"}


# ------------------------------------------------- Markdown / text chunking

@pytest.mark.parametrize("max_size", SIZES)
@pytest.mark.parametrize("text", [MD_LORA, MD_LONG, MD_CRLF])
def test_markdown_chunks_respect_the_size_and_cover_the_file(
        text: str, max_size: int) -> None:
    """Markdown chunks are exact spans no longer than max_chunk_size."""
    chunks = markdown_chunker.chunk(text, "data/raw/x/doc.md", max_size, 1)
    assert_valid_chunks(chunks, text, max_size)
    assert all(c.kind == "markdown" for c in chunks)


def test_markdown_is_cut_at_headings_with_heading_path_context() -> None:
    """Sections start at headings and remember their heading path."""
    chunks = markdown_chunker.chunk(MD_LORA, "data/raw/x/lora.md", 2000, 1)
    firsts = [MD_LORA[c.first_character_index:].split("\n", 1)[0]
              for c in chunks]
    assert firsts == ["# LoRA Adapters", "## Serving adapters",
                      "## Dynamic loading"]
    assert chunks[2].context == "LoRA Adapters > Dynamic loading"


def test_headings_inside_code_fences_are_ignored() -> None:
    """A ``#`` comment inside a fenced block is not a heading."""
    chunks = markdown_chunker.chunk(MD_LORA, "data/raw/x/lora.md", 2000, 1)
    serving = MD_LORA[chunks[1].first_character_index:
                      chunks[1].last_character_index]
    assert "# this line is a shell comment" in serving
    assert "vllm serve base-model" in serving


def test_small_markdown_sections_are_merged() -> None:
    """Small neighbouring sections are merged up to the merge target."""
    chunks = markdown_chunker.chunk(MD_LORA, "data/raw/x/lora.md", 2000, 1000)
    assert len(chunks) == 1


def test_long_sections_are_split_on_paragraphs() -> None:
    """A section over the limit is cut between paragraphs."""
    chunks = markdown_chunker.chunk(MD_LONG, "data/raw/x/long.md", 500, 1)
    assert len(chunks) > 1
    for chunk in chunks[1:]:
        assert MD_LONG[chunk.first_character_index:].startswith("Paragraph")


def test_crlf_offsets_are_exact() -> None:
    """Offsets index the file as stored, CRLF included."""
    chunks = markdown_chunker.chunk(MD_CRLF, "data/raw/x/win.md", 2000, 1)
    second = chunks[-1]
    assert MD_CRLF[second.first_character_index:].startswith(
        "## Second heading\r\n")


@pytest.mark.parametrize("max_size", SIZES)
def test_text_chunking(max_size: int) -> None:
    """Plain text is cut on paragraphs within the size limit."""
    chunks = text_chunker.chunk(MD_LONG, "data/raw/x/notes.txt",
                                max_size, max_size)
    assert_valid_chunks(chunks, MD_LONG, max_size)


def test_split_by_lines_cuts_overlong_lines() -> None:
    """A single line longer than the limit is hard-sliced."""
    text = "x" * 4500 + "\nshort\n"
    spans = span_splitter.split_by_lines(text, 0, len(text), 2000)
    assert all(end - start <= 2000 for start, end in spans)
    assert "".join(text[s:e] for s, e in spans) == text


def test_two_distinct_strategies_are_dispatched_by_extension() -> None:
    """``.py`` goes to the code chunker, ``.md`` to the Markdown one."""
    chunks = file_chunker.chunk(PY_SCHEDULER, "a.py", 2000, 1)
    assert {c.kind for c in chunks} == {"python"}
    assert {c.kind for c in file_chunker.chunk(MD_LORA, "a.md", 2000, 1)} == \
        {"markdown"}
    chunks = file_chunker.chunk("hello\n", "a.txt", 2000, 1)
    assert {c.kind for c in chunks} == {"text"}


# ------------------------------------------------------- corpus & tokenizer

def test_discover_files_keeps_text_sources_and_skips_vcs(
        mini_corpus: MiniCorpus) -> None:
    """Python / Markdown / text files are found; .git and binaries not."""
    found = corpus_repository.discover(mini_corpus.raw_dir)
    assert "data/raw/proj/vllm/core/scheduler.py" in found
    assert "data/raw/proj/docs/features/lora.md" in found
    assert "data/raw/proj/notes.txt" in found
    assert not any(".git/" in path or path.endswith(".cu")
                   for path in found)
    assert all(path.startswith("data/raw/") for path in found)


def test_reading_keeps_newlines_untranslated(
        mini_corpus: MiniCorpus) -> None:
    """Text is read byte-exact so offsets match the grader's."""
    path = "data/raw/proj/docs/windows.md"
    text, digest = corpus_repository.read_with_digest(path)
    assert text == corpus_repository.read_text(path) == MD_CRLF
    assert len(digest) == 40


def test_tokenizer_splits_identifiers() -> None:
    """Identifiers are kept whole and split into their words."""
    tokens = tokenizer.tokenize("PagedAttentionBackend max_num_seqs")
    for term in ("pagedattentionbackend", "paged", "attention", "backend",
                 "max_num_seq", "max", "num", "seq"):
        assert term in tokens
    assert tokenizer.split_identifier("HTTPServer") == ["http", "server"]


def test_tokenizer_drops_stopwords_and_stems() -> None:
    """Stop words disappear, plurals are folded."""
    words = tokenizer.tokenize("How do the adapters work?")
    assert words == ["adapter", "work"]
    assert tokenizer.stem("policies") == "policy"
    assert tokenizer.stem("class") == "class"


# --------------------------------------------------------------- BM25 index

def test_bm25_weights_follow_the_okapi_formula() -> None:
    """Scores match idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b dl/avgdl))."""
    docs = [["lora", "adapter", "lora"], ["adapter", "server"], ["kernel"]]
    index = BM25Index(k1=1.2, b=0.75)
    index.build(docs)
    avgdl = sum(len(d) for d in docs) / len(docs)

    def expected(doc: List[str], term: str) -> float:
        df = sum(term in d for d in docs)
        idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
        tf = doc.count(term)
        norm = 1.2 * (1 - 0.75 + 0.75 * len(doc) / avgdl)
        return idf * tf * 2.2 / (tf + norm)

    scores = index.score(["lora", "adapter"])
    for i, doc in enumerate(docs):
        want = expected(doc, "lora") + expected(doc, "adapter")
        assert scores[i] == pytest.approx(want, rel=1e-5)


def test_bm25_unknown_terms_score_zero() -> None:
    """A query with no known term scores every document 0."""
    index = BM25Index()
    index.build([["a1"], ["b2"]])
    assert not index.score(["zzz"]).any()


def test_bm25_save_and_load_roundtrip(tmp_path: os.PathLike[str]) -> None:
    """A persisted index scores exactly like the original."""
    index = BM25Index()
    index.build([["lora", "adapter"], ["kernel", "cuda", "kernel"]])
    bm25_repository.save(index, str(tmp_path))
    loaded = bm25_repository.load(str(tmp_path), with_counts=True)
    assert np.allclose(loaded.score(["kernel"]), index.score(["kernel"]))
    assert loaded.counts.shape == index.counts.shape


# ------------------------------------------------------ persisted index

def test_index_is_persisted_under_the_index_directory(
        mini_corpus: MiniCorpus, mini_index: IndexReport) -> None:
    """Chunks, BM25 matrices, vocabulary and metadata are written."""
    files = set(os.listdir(mini_corpus.native_dir))
    assert {"chunks.jsonl", "bm25_weights.npz", "bm25_counts.npz",
            "bm25_vocab.json", "meta.json", "manifest.json"} <= files
    chunks = chunk_repository.load(mini_corpus.native_dir)
    assert len(chunks) == mini_index.n_chunks > 0
    assert {c.file_path for c in chunks} <= set(mini_corpus.files)


@pytest.mark.parametrize("max_size", SIZES)
def test_max_chunk_size_is_honoured_by_the_indexer(
        mini_corpus: MiniCorpus, max_size: int) -> None:
    """Every indexed chunk is at most --max_chunk_size characters."""
    native_indexer.build(mini_corpus.raw_dir, mini_corpus.native_dir, max_size)
    for chunk in chunk_repository.load(mini_corpus.native_dir):
        size = chunk.last_character_index - chunk.first_character_index
        assert 0 < size <= max_size
        text = corpus_repository.read_text(chunk.file_path)
        assert chunk.last_character_index <= len(text)


def test_index_command_defaults_to_2000_characters(
        mini_corpus: MiniCorpus, cli: Callable[..., CLIResult]) -> None:
    """``index`` works with its defaults and reports the chunk count."""
    result = cli("index")
    assert result.code == 0, result.err
    assert "Ingestion complete" in result.out
    meta = (mini_corpus.root / "data/processed/native/meta.json").read_text()
    assert '"max_chunk_size": 2000' in meta


@pytest.mark.parametrize("value", ["2001", "5000", "0", "-5", "abc", "1.5"])
def test_index_command_rejects_invalid_chunk_sizes(
        mini_corpus: MiniCorpus, cli: Callable[..., CLIResult],
        value: str) -> None:
    """Chunk sizes above 2000 (or not positive integers) are refused."""
    result = cli("index", "--max_chunk_size", value)
    assert result.code == 1
    assert result.err.startswith("Error:") and "max_chunk_size" in result.err


def test_index_command_reports_a_missing_corpus(
        mini_corpus: MiniCorpus, cli: Callable[..., CLIResult]) -> None:
    """A missing raw directory is an error message, not a crash."""
    result = cli("index", "--raw_dir", "data/nope")
    assert result.code == 1 and "corpus directory not found" in result.err


# ----------------------------------------------------------- real corpus

@requires_corpus
def test_whole_corpus_is_indexed_in_under_5_minutes(
        corpus_index: CorpusIndex) -> None:
    """Indexing the vLLM corpus takes at most 5 minutes."""
    assert corpus_index.build_seconds < 300


@requires_corpus
def test_whole_corpus_chunks_are_valid(corpus_index: CorpusIndex,
                                       in_project_root: object) -> None:
    """Both strategies are used and no chunk exceeds 2000 characters."""
    chunks = chunk_repository.load(corpus_index.native_dir)
    assert len(chunks) > 10000
    assert {"python", "markdown"} <= {c.kind for c in chunks}
    assert max(c.last_character_index - c.first_character_index
               for c in chunks) <= 2000
    assert all(c.file_path.startswith("data/raw/vllm-0.10.1/")
               for c in chunks)
