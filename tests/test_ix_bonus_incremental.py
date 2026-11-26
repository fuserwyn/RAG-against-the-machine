"""IX - Bonus 3: incremental indexing (re-index only changed files)."""

import os
from typing import Callable, List

import numpy as np
import pytest

from src.models.index_build import IndexReport
from src.repositories.chunk_repository import chunk_repository
from src.bonus.semantic.embedding_repository import embedding_repository
from src.services.indexer import native_indexer
from src.services.retriever import LexicalRetriever
from src.bonus.semantic.semantic_indexer import semantic_indexer

from .helpers import CLIResult, EmbedCalls, MiniCorpus

LORA_MD = "data/raw/proj/docs/features/lora.md"
NEW_DOC = "data/raw/proj/docs/new_feature.md"
QUERIES = ["load lora adapter", "scheduler budget method",
           "zephyrquokka knob", "speculative decoding draft",
           "quantization int8 kernel", "flamingoparrot"]


def snapshot(corpus: MiniCorpus) -> List[object]:
    """Chunks and search results of the current index."""
    retriever = LexicalRetriever(corpus.native_dir)
    return [chunk_repository.load(corpus.native_dir)] + \
        [retriever.search(q, 5) for q in QUERIES] + \
        [[round(float(x), 5) for x in retriever.bm25.score(["lora"])]]


def full_rebuild_snapshot(corpus: MiniCorpus) -> List[object]:
    """Snapshot of a from-scratch build of the current corpus."""
    native_indexer.build(corpus.raw_dir, corpus.native_dir, 2000)
    return snapshot(corpus)


def test_unchanged_corpus_reuses_every_file(mini_corpus: MiniCorpus,
                                            mini_index: IndexReport) -> None:
    """Nothing changed: nothing is re-chunked, the index is identical."""
    before = snapshot(mini_corpus)
    report = native_indexer.build(mini_corpus.raw_dir,
                                  mini_corpus.native_dir, 2000,
                                  incremental=True)
    assert report.incremental and report.indexed_files == []
    assert len(report.reused_files) == mini_index.n_files
    assert report.previous_build_id == mini_index.build_id
    assert report.build_id != mini_index.build_id
    assert snapshot(mini_corpus) == before


def test_only_the_modified_file_is_reindexed(
        mini_corpus: MiniCorpus, mini_index: IndexReport) -> None:
    """Editing one file re-chunks that file only, and the result is the
    same as a full rebuild."""
    mini_corpus.write(LORA_MD, mini_corpus.files[LORA_MD] +
                      "\n## Flamingoparrot\n\nA brand new section.\n")
    report = native_indexer.build(mini_corpus.raw_dir,
                                  mini_corpus.native_dir, 2000,
                                  incremental=True)
    assert report.indexed_files == [LORA_MD]
    assert LORA_MD not in report.reused_files
    incremental = snapshot(mini_corpus)
    retriever = LexicalRetriever(mini_corpus.native_dir)
    assert retriever.search("flamingoparrot", 1)[0].file_path == LORA_MD
    assert incremental == full_rebuild_snapshot(mini_corpus)


def test_added_and_removed_files(mini_corpus: MiniCorpus,
                                 mini_index: IndexReport) -> None:
    """New files are indexed, deleted files disappear from the index."""
    mini_corpus.write(NEW_DOC, "# New feature\n\nFlamingoparrot mode.\n")
    os.remove(mini_corpus.root / "data/raw/proj/notes.txt")
    report = native_indexer.build(mini_corpus.raw_dir,
                                  mini_corpus.native_dir, 2000,
                                  incremental=True)
    assert report.indexed_files == [NEW_DOC]
    assert report.removed_files == ["data/raw/proj/notes.txt"]
    chunks = chunk_repository.load(mini_corpus.native_dir)
    paths = {c.file_path for c in chunks}
    assert NEW_DOC in paths and "data/raw/proj/notes.txt" not in paths
    assert snapshot(mini_corpus) == full_rebuild_snapshot(mini_corpus)


@pytest.mark.parametrize("change", ["size", "missing_meta", "corrupted"])
def test_incompatible_previous_index_falls_back_to_a_full_build(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        change: str) -> None:
    """Different settings or a damaged index trigger a full rebuild."""
    size = 2000
    if change == "size":
        size = 500
    elif change == "missing_meta":
        os.remove(os.path.join(mini_corpus.native_dir, "meta.json"))
    else:
        with open(os.path.join(mini_corpus.native_dir, "manifest.json"),
                  "w", encoding="utf-8") as handle:
            handle.write("{oops")
    report = native_indexer.build(mini_corpus.raw_dir,
                                  mini_corpus.native_dir, size,
                                  incremental=True)
    assert not report.incremental
    assert len(report.indexed_files) == mini_index.n_files


def test_embeddings_of_unchanged_chunks_are_reused(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        fake_embedder: EmbedCalls) -> None:
    """After an edit, only the edited file's chunks are embedded again."""
    semantic_indexer.build(mini_corpus.native_dir, mini_index)
    first = len(fake_embedder.texts)
    mini_corpus.write(NEW_DOC, "# New feature\n\nFlamingoparrot mode.\n")
    report = native_indexer.build(mini_corpus.raw_dir,
                                  mini_corpus.native_dir, 2000,
                                  incremental=True)
    embedded = semantic_indexer.build(mini_corpus.native_dir, report)
    assert embedded == 1
    assert fake_embedder.texts[first:] == \
        ["New feature\n# New feature\n\nFlamingoparrot mode.\n"]
    incremental = np.load(os.path.join(mini_corpus.native_dir,
                                       embedding_repository.vectors_file))
    semantic_indexer.build(mini_corpus.native_dir, None)
    full = np.load(os.path.join(mini_corpus.native_dir,
                                embedding_repository.vectors_file))
    assert np.allclose(incremental, full)


def test_incremental_flag_from_the_cli(mini_corpus: MiniCorpus,
                                       mini_index: IndexReport,
                                       fake_embedder: EmbedCalls,
                                       cli: Callable[..., CLIResult]) -> None:
    """``index --incremental`` reports what it did and keeps embeddings
    in sync."""
    cli("index", "--semantic")
    mini_corpus.write(LORA_MD, mini_corpus.files[LORA_MD] + "\nMore.\n")
    result = cli("index", "--incremental")
    assert result.code == 0, result.err
    assert "incremental update: 1 file(s) re-indexed, 6 unchanged, 0 " \
        "removed" in result.out
    assert "chunk(s) embedded" in result.out
    found = cli("--mode", "semantic", "search", "lora")
    assert found.code == 0, found.err


def test_incremental_is_native_only(mini_corpus: MiniCorpus,
                                    cli: Callable[..., CLIResult]) -> None:
    """The LangChain backend refuses --incremental cleanly."""
    result = cli("--backend", "langchain", "index", "--incremental")
    assert result.code == 1 and "native backend" in result.err
