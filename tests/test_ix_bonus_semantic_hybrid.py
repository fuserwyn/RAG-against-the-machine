"""IX - Bonuses 1 and 2: semantic embeddings and hybrid retrieval."""

import json
import os
from typing import Callable

import numpy as np
import pytest

from src.models.index_build import IndexReport
from src.bonus.semantic.embedding_repository import embedding_repository
from src.bonus.semantic.embedder import Embedder
from src.services.indexer import native_indexer
from src.bonus.hybrid.rank_fusion import rank_fusion
from src.bonus.hybrid.hybrid_retriever import HybridRetriever
from src.bonus.semantic.semantic_retriever import SemanticRetriever
from src.bonus.semantic.semantic_indexer import (SemanticIndexer,
                                                 semantic_indexer)

from .helpers import (FAKE_DIM, MINILM, CLIResult, EmbedCalls, MiniCorpus,
                      requires_model)

LORA_MD = "data/raw/proj/docs/features/lora.md"


# --------------------------------------------------- 1. semantic embeddings

def test_semantic_index_is_stored_next_to_the_lexical_one(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        fake_embedder: EmbedCalls) -> None:
    """One normalized vector per chunk, tied to the lexical build."""
    assert semantic_indexer.build(mini_corpus.native_dir, mini_index) == \
        mini_index.n_chunks
    vectors = np.load(os.path.join(mini_corpus.native_dir,
                                   embedding_repository.vectors_file))
    assert vectors.shape == (mini_index.n_chunks, FAKE_DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    meta_path = os.path.join(mini_corpus.native_dir,
                             embedding_repository.meta_file)
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    assert meta["build_id"] == mini_index.build_id
    assert meta["model"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_embedding_text_includes_the_context() -> None:
    """The encoder sees the class / heading context and the body."""
    from src.models.chunk import Chunk

    chunk = Chunk(file_path="a.md", first_character_index=2,
                  last_character_index=7, kind="markdown",
                  context="LoRA > Serving")
    assert SemanticIndexer.embedding_text(chunk, "xxhelloyy") == \
        "LoRA > Serving\nhello"


def test_semantic_search_ranks_by_cosine_similarity(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        fake_embedder: EmbedCalls) -> None:
    """The closest chunk in vector space is returned first."""
    semantic_indexer.build(mini_corpus.native_dir, mini_index)
    retriever = SemanticRetriever(mini_corpus.native_dir)
    query = "zephyrquokka tuning knob paragraph"
    sources = retriever.search(query, 3)
    assert sources[0].file_path == "data/raw/proj/docs/zephyr.md"
    scores = retriever.scores(query)
    ranked = [scores[i] for i in retriever.ranking(query, 5)]
    assert ranked == sorted(ranked, reverse=True)
    assert retriever.search("", 3) == [] and retriever.search("x", 0) == []


def test_stale_embeddings_are_detected(mini_corpus: MiniCorpus,
                                       mini_index: IndexReport,
                                       fake_embedder: EmbedCalls) -> None:
    """Embeddings of an older build are refused, not misused."""
    semantic_indexer.build(mini_corpus.native_dir, mini_index)
    native_indexer.build(mini_corpus.raw_dir, mini_corpus.native_dir, 2000)
    with pytest.raises(ValueError, match="re-run 'index --semantic'"):
        SemanticRetriever(mini_corpus.native_dir)


def test_semantic_mode_from_the_cli(mini_corpus: MiniCorpus,
                                    fake_embedder: EmbedCalls,
                                    cli: Callable[..., CLIResult]) -> None:
    """``index --semantic`` then ``--mode semantic search``."""
    result = cli("index", "--semantic")
    assert result.code == 0, result.err
    assert "chunk(s) embedded" in result.out
    found = cli("--mode", "semantic", "search", "zephyrquokka", "--k", "2")
    assert found.code == 0, found.err
    assert json.loads(found.out)["retrieved_sources"]


def test_semantic_mode_without_embeddings_is_a_clean_error(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        cli: Callable[..., CLIResult]) -> None:
    """Forgetting ``--semantic`` at indexing time is reported."""
    result = cli("--mode", "semantic", "search", "lora")
    assert result.code == 1 and "index --semantic" in result.err


def test_plain_reindex_removes_stale_embeddings(
        mini_corpus: MiniCorpus, fake_embedder: EmbedCalls,
        cli: Callable[..., CLIResult]) -> None:
    """A full lexical rebuild deletes embeddings it would invalidate."""
    cli("index", "--semantic")
    assert cli("index").code == 0
    assert not os.path.exists(os.path.join(mini_corpus.native_dir,
                                           embedding_repository.vectors_file))


@requires_model(MINILM)
def test_minilm_embeddings(mini_corpus: MiniCorpus) -> None:
    """The real all-MiniLM-L6-v2 gives 384-d unit vectors and sensible
    similarities."""
    embedder = Embedder(MINILM)
    vectors = embedder.encode(["How do I serve LoRA adapters?",
                               "Serving low-rank adapters with vLLM",
                               "The weather is sunny in Paris"])
    assert vectors.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]


# ------------------------------------------------------ 2. hybrid retrieval

def test_rrf_fuse_rewards_agreement() -> None:
    """A document ranked well by both lists wins."""
    fused = rank_fusion.fuse([([1, 2, 3], 1.0), ([3, 2, 4], 1.0)], rrf_k=60)
    assert fused[0] in (2, 3) and set(fused) == {1, 2, 3, 4}
    assert fused.index(2) < fused.index(4)


def test_rrf_fuse_formula_and_weights() -> None:
    """fused(d) = sum weight / (rrf_k + rank); weights tilt the order."""
    assert rank_fusion.fuse([([7], 1.0), ([8], 0.5)], rrf_k=10) == [7, 8]
    assert rank_fusion.fuse([([7], 0.5), ([8], 1.0)], rrf_k=10) == [8, 7]
    assert rank_fusion.fuse([([1, 2], 1.0), ([2, 1], 1.0)]) == [1, 2]
    assert rank_fusion.fuse([]) == []


def test_hybrid_default_weights_favour_bm25() -> None:
    """MiniLM is weaker than BM25 here, so it gets the smaller weight."""
    assert HybridRetriever.LEXICAL_WEIGHT > HybridRetriever.SEMANTIC_WEIGHT > 0


def test_hybrid_retriever_combines_both_rankings(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        fake_embedder: EmbedCalls) -> None:
    """Results come from the union of the lexical and semantic lists."""
    semantic_indexer.build(mini_corpus.native_dir, mini_index)
    hybrid = HybridRetriever(mini_corpus.native_dir)
    query = "load LoRA adapter at runtime"
    sources = hybrid.search(query, 3)
    assert sources[0].file_path == LORA_MD
    union = set(hybrid.lexical.ranking(query, 100)) | \
        set(hybrid.semantic.ranking(query, 100))
    chunks = hybrid.lexical.chunks
    allowed = {(chunks[i].file_path, chunks[i].first_character_index)
               for i in union}
    assert {(s.file_path, s.first_character_index) for s in sources} <= \
        allowed
    assert hybrid.search("  ", 3) == []


def test_hybrid_matches_lexical_when_semantic_weight_is_zero(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        fake_embedder: EmbedCalls) -> None:
    """With weight 0 for embeddings, hybrid == BM25 order."""
    semantic_indexer.build(mini_corpus.native_dir, mini_index)
    hybrid = HybridRetriever(mini_corpus.native_dir, semantic_weight=0.0)
    query = "scheduler budget planning method"
    assert hybrid.search(query, 4) == hybrid.lexical.search(query, 4)


def test_hybrid_mode_from_the_cli(mini_corpus: MiniCorpus,
                                  fake_embedder: EmbedCalls,
                                  cli: Callable[..., CLIResult]) -> None:
    """``--mode hybrid`` works end to end and needs the native backend."""
    cli("index", "--semantic")
    found = cli("--mode", "hybrid", "search", "lora adapter runtime")
    assert found.code == 0, found.err
    assert json.loads(found.out)["retrieved_sources"][0]["file_path"] == \
        LORA_MD
    refused = cli("--backend", "langchain", "--mode", "hybrid", "search",
                  "x")
    assert refused.code == 1 and "native backend" in refused.err
