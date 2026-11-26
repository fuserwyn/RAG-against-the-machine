"""IX - Bonus 4: caching of the index and of query results."""

import os
from pathlib import Path
from typing import Callable, List

from src.models.index_build import IndexReport
from src.bonus.cache.cache_repository import CacheRepository
from src.schemas import MinimalSource
from src.bonus.cache.cached_search import CachedSearch
from src.services.rag_service import RagService
from src.views.cli import RagCLI

from .helpers import CLIResult, GenerateCalls, MiniCorpus


class CountingBackend:
    """A retriever that counts its searches."""

    def __init__(self) -> None:
        """Start with no search."""
        self.calls = 0

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """Return one fake source per query."""
        self.calls += 1
        return [MinimalSource(file_path=f"{query}.md",
                              first_character_index=0,
                              last_character_index=k)]


def test_query_cache_get_set_and_persistence(tmp_path: Path) -> None:
    """Values survive a new cache instance (they are on disk)."""
    path = str(tmp_path / CacheRepository.file_name)
    cache = CacheRepository(path)
    assert cache.get("k") is None
    cache.set("k", "v")
    assert cache.get("k") == "v"
    assert CacheRepository(path).get("k") == "v"


def test_query_cache_memory_lru(tmp_path: Path) -> None:
    """The in-memory layer keeps only the most recent entries."""
    cache = CacheRepository(str(tmp_path / CacheRepository.file_name),
                            memory_size=2)
    for key in ("a", "b", "c"):
        cache.set(key, key.upper())
    assert list(cache._memory) == ["b", "c"]
    assert cache.get("a") == "A"


def test_query_cache_clear(tmp_path: Path) -> None:
    """clear() forgets everything and deletes the file."""
    path = tmp_path / CacheRepository.file_name
    cache = CacheRepository(str(path))
    cache.set("k", "v")
    cache.clear()
    assert not path.exists() and cache.get("k") is None


def test_query_cache_is_best_effort(tmp_path: Path) -> None:
    """An unusable location behaves as an empty cache, never crashes."""
    cache = CacheRepository(
        str(tmp_path / "missing_dir" / CacheRepository.file_name))
    cache.set("k", "v")
    assert CacheRepository(cache.path).get("k") is None
    corrupt = tmp_path / "corrupt.sqlite"
    corrupt.write_text("this is not a database")
    assert CacheRepository(str(corrupt)).get("k") is None


def test_make_key_is_stable_and_discriminating() -> None:
    """Same parts, same key; any different part, different key."""
    key = CacheRepository.make_key("search", "native", "lora", 5)
    assert key == CacheRepository.make_key("search", "native", "lora", 5)
    assert key != CacheRepository.make_key("search", "native", "lora", 6)
    assert len(key) == 64


def test_cached_search_skips_the_backend_on_repeat(tmp_path: Path) -> None:
    """A repeated query does not reach the retriever."""
    backend = CountingBackend()
    cache = CacheRepository(str(tmp_path / CacheRepository.file_name))
    cached = CachedSearch(lambda: backend, cache,
                          ["native", "lexical", "build"])
    first = cached.search("lora", 3)
    assert cached.search("lora", 3) == first
    assert backend.calls == 1
    cached.search("lora", 4)
    assert backend.calls == 2


def test_cold_start_hit_does_not_load_the_index(tmp_path: Path) -> None:
    """A fresh process answers a cached query without loading anything."""
    path = str(tmp_path / CacheRepository.file_name)
    CachedSearch(CountingBackend, CacheRepository(path), ["s"]).search("q", 2)

    def must_not_load() -> CountingBackend:
        raise AssertionError("the index was loaded on a cache hit")

    fresh = CachedSearch(must_not_load, CacheRepository(path), ["s"])
    assert fresh.search("q", 2)[0].file_path == "q.md"
    assert not fresh.loaded


def test_cli_search_cache_survives_processes(
        mini_corpus: MiniCorpus, mini_index: IndexReport) -> None:
    """Two CLI instances: the second answers from the cache."""
    first = RagService()
    expected = first.search("lora adapter", 3)
    assert os.path.isfile(os.path.join(mini_corpus.native_dir,
                                       CacheRepository.file_name))
    second = RagService()
    assert second.search("lora adapter", 3) == expected
    assert not second.retriever().loaded


def test_cache_can_be_disabled(mini_corpus: MiniCorpus,
                               mini_index: IndexReport,
                               cli: Callable[..., CLIResult]) -> None:
    """``--cache False`` never touches the cache file."""
    result = cli("--cache", "False", "search", "lora")
    assert result.code == 0, result.err
    assert not os.path.exists(os.path.join(mini_corpus.native_dir,
                                           CacheRepository.file_name))


def test_reindexing_invalidates_the_cache(mini_corpus: MiniCorpus,
                                          cli: Callable[..., CLIResult]
                                          ) -> None:
    """Results cached for an old index are never served for a new one."""
    cli("index")
    before = cli("search", "flamingoparrot").out
    assert '"retrieved_sources": []' in before
    mini_corpus.write("data/raw/proj/docs/new.md",
                      "# New\n\nFlamingoparrot appears here.\n")
    cli("index", "--incremental")
    after = cli("search", "flamingoparrot").out
    assert "data/raw/proj/docs/new.md" in after


def test_answers_are_cached(mini_corpus: MiniCorpus,
                            mini_index: IndexReport,
                            fake_generator: GenerateCalls) -> None:
    """A repeated question is answered without loading the LLM again."""
    first = RagCLI().answer("load lora adapter", 2)
    assert fake_generator.loads == 1
    second = RagCLI().answer("load lora adapter", 2)
    assert second == first
    assert fake_generator.loads == 1
    RagCLI().answer("load lora adapter", 2, model_name="other/model")
    assert fake_generator.loads == 2


def test_answer_dataset_reuses_cached_answers(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        fake_generator: GenerateCalls, cli: Callable[..., CLIResult]
        ) -> None:
    """Re-running answer_dataset does not load the model again."""
    dataset = mini_corpus.root / "d.json"
    dataset.write_text('{"rag_questions": [{"question": "lora adapter"}]}')
    cli("search_dataset", "--dataset_path", str(dataset),
        "--save_directory", "s")
    for _ in range(2):
        assert cli("answer_dataset", "--student_search_results_path",
                   "s/d.json", "--save_directory", "a").code == 0
    assert fake_generator.loads == 1
