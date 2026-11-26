"""The vector-database backend (Qdrant, embedded mode)."""

import json
import os
from typing import Callable

import pytest

from src.repositories.corpus_repository import corpus_repository
from src.extra.qdrant.qdrant_repository import qdrant_repository
from src.extra.qdrant.retriever import QdrantRetriever

from .helpers import FAKE_DIM, CLIResult, EmbedCalls, MiniCorpus


def test_qdrant_backend_end_to_end(mini_corpus: MiniCorpus,
                                   fake_embedder: EmbedCalls,
                                   cli: Callable[..., CLIResult]) -> None:
    """``--backend qdrant`` embeds every chunk and searches by vector."""
    built = cli("--backend", "qdrant", "index", "--max_chunk_size", "500")
    assert built.code == 0, built.err
    assert "embedded into Qdrant" in built.out
    meta = json.loads(
        corpus_repository.read_text("data/processed/qdrant/meta.json"))
    assert meta["max_chunk_size"] == 500 and meta["dim"] == FAKE_DIM
    assert meta["n_chunks"] == len(fake_embedder.texts)
    assert os.path.isdir(os.path.join("data/processed/qdrant",
                                      qdrant_repository.storage_dir))

    found = cli("--backend", "qdrant", "--cache", "False", "search",
                "load a LoRA adapter at runtime", "--k", "3")
    assert found.code == 0, found.err
    sources = json.loads(found.out)["retrieved_sources"]
    assert len(sources) == 3
    paths = [source["file_path"] for source in sources]
    assert "data/raw/proj/docs/features/lora.md" in paths
    for source in sources:
        text = corpus_repository.read_text(source["file_path"])
        assert 0 < source["last_character_index"] - \
            source["first_character_index"] <= 500
        assert source["last_character_index"] <= len(text)


def test_qdrant_scores_are_cosine(mini_corpus: MiniCorpus,
                                  fake_embedder: EmbedCalls,
                                  cli: Callable[..., CLIResult]) -> None:
    """Scores are cosine similarities, best first, payload has the span."""
    assert cli("--backend", "qdrant", "index").code == 0
    retriever = QdrantRetriever("data/processed/qdrant")
    try:
        hits = retriever.scored("lora adapter", 4)
        scores = [score for _, score, _ in hits]
        assert scores == sorted(scores, reverse=True)
        assert all(-1.0001 <= score <= 1.0001 for score in scores)
        assert all("kind" in payload for _, _, payload in hits)
        assert retriever.scored("lora", 0) == []
        assert retriever.scored("   ", 3) == []
    finally:
        retriever.client.close()


def test_qdrant_errors_are_clean(mini_corpus: MiniCorpus,
                                 fake_embedder: EmbedCalls,
                                 cli: Callable[..., CLIResult]) -> None:
    """Missing index, k=0 and semantic mode give an error, no traceback."""
    missing = cli("--backend", "qdrant", "search", "lora")
    assert missing.code == 1 and "qdrant index" in missing.err
    assert "Traceback" not in missing.err
    assert cli("--backend", "qdrant", "index").code == 0
    zero = cli("--backend", "qdrant", "search", "lora", "--k", "0")
    assert zero.code == 1 and "Traceback" not in zero.err
    mode = cli("--backend", "qdrant", "--mode", "semantic", "search", "x")
    assert mode.code == 1 and "native" in mode.err


@pytest.mark.parametrize("flag", ["--semantic", "--incremental"])
def test_qdrant_rejects_native_only_flags(
        mini_corpus: MiniCorpus, cli: Callable[..., CLIResult],
        flag: str) -> None:
    """Bonus flags stay native-only."""
    result = cli("--backend", "qdrant", "index", flag)
    assert result.code == 1 and "native" in result.err
