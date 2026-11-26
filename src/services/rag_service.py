"""The RAG pipeline behind both views (CLI and HTTP API).

    index            corpus -> chunks -> index       (one backend)
    search           question -> top-k sources       (cached)
    answer           sources -> Qwen3 answer         (cached)
    *_dataset        the same for a whole JSON file  (tqdm)
    evaluate         recall@k against a ground truth
"""

import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from pydantic import ValidationError
from tqdm import tqdm

from ..bonus.cache.cache_repository import CacheRepository
from ..bonus.cache.cached_search import CachedSearch
from ..bonus.semantic.embedding_repository import embedding_repository
from ..bonus.semantic.semantic_indexer import semantic_indexer
from ..config import settings
from ..errors import RagError
from ..extra.langchain.indexer import langchain_indexer
from ..extra.qdrant.indexer import qdrant_indexer
from ..models.evaluation_report import EvaluationReport
from ..repositories.json_repository import json_repository
from ..repositories.meta_repository import meta_repository
from ..schemas.question import RagDataset
from ..schemas.search import (MinimalAnswer, MinimalSearchResults,
                              StudentSearchResults,
                              StudentSearchResultsAndAnswer)
from ..schemas.source import MinimalSource
from .evaluator import evaluator
from .generator import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, generator_provider
from .indexer import native_indexer
from .retriever_factory import retriever_factory
from .search_backend import SearchBackend
from .validator import validator


class RagService:
    """One backend + mode over one index directory."""

    def __init__(self, index_dir: str = settings.index_dir,
                 backend: str = "native", mode: str = "lexical",
                 cache: bool = True) -> None:
        """Choose the backend, retrieval mode and index directory.

        Args:
            index_dir: Root index directory; each backend uses its own
                sub-directory (``<index_dir>/<backend>``).
            backend: ``native``, ``langchain`` or ``qdrant``.
            mode: ``lexical``, ``semantic`` or ``hybrid`` (the last two
                need the native backend and ``index --semantic``).
            cache: Cache search results and answers.
        """
        self.backend = validator.choice(backend, "backend",
                                        settings.backends)
        self.mode = validator.choice(mode, "mode", settings.modes)
        if self.mode != "lexical" and self.backend != "native":
            raise RagError(f"mode {self.mode!r} is only available with the "
                           "native backend")
        self.index_dir = os.path.join(str(index_dir), self.backend)
        self._cache: Optional[CacheRepository] = (
            CacheRepository.in_index(self.index_dir)
            if validator.flag(cache, "cache") else None)
        self._retriever: Optional[CachedSearch] = None
        self._generators: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._generate_lock = threading.Lock()

    def info(self) -> Dict[str, str]:
        """Backend, mode and index directory in use."""
        return {"backend": self.backend, "mode": self.mode,
                "index_dir": self.index_dir}

    # ---- indexing ------------------------------------------------------
    def index(self, max_chunk_size: Any, raw_dir: Any, semantic: Any,
              incremental: Any) -> str:
        """Build the index of the selected backend; return a summary."""
        semantic = validator.flag(semantic, "semantic")
        incremental = validator.flag(incremental, "incremental")
        if (semantic or incremental) and self.backend != "native":
            raise RagError("--semantic and --incremental are only "
                           "available with the native backend")
        size = validator.chunk_size(max_chunk_size)
        if not isinstance(raw_dir, str) or not os.path.isdir(raw_dir):
            raise RagError(f"corpus directory not found: {raw_dir}")
        start = time.perf_counter()
        try:
            summary = self._build(raw_dir, size, semantic, incremental)
        except ImportError as err:
            raise RagError(f"missing dependency ({err}); run "
                           "'uv sync'") from err
        except OSError as err:
            raise RagError(str(err)) from err
        CacheRepository.in_index(self.index_dir).clear()
        elapsed = time.perf_counter() - start
        return (f"Ingestion complete ({self.backend})! {summary} in "
                f"{elapsed:.1f}s. Indices saved under {self.index_dir}/")

    def _build(self, raw_dir: str, size: int, semantic: bool,
               incremental: bool) -> str:
        """Run the indexer of the backend."""
        if self.backend == "langchain":
            n_chunks = langchain_indexer.build(raw_dir, self.index_dir, size)
            return f"{n_chunks} chunks indexed"
        if self.backend == "qdrant":
            n_chunks = qdrant_indexer.build(raw_dir, self.index_dir, size)
            return f"{n_chunks} chunks embedded into Qdrant"
        report = native_indexer.build(raw_dir, self.index_dir, size,
                                      incremental=incremental)
        summary = report.summary()
        if semantic or (incremental
                        and embedding_repository.exists(self.index_dir)):
            embedded = semantic_indexer.build(self.index_dir, report)
            summary += f", {embedded} chunk(s) embedded"
        else:
            embedding_repository.remove(self.index_dir)
        return summary

    # ---- retrieval -----------------------------------------------------
    def retriever(self) -> CachedSearch:
        """The (cached, lazily loaded) retriever of this backend / mode."""
        with self._lock:
            if self._retriever is None:
                scope = [self.backend, self.mode,
                         str(meta_repository.build_id(self.index_dir))]
                self._retriever = CachedSearch(self._load_retriever,
                                               self._index_cache(), scope)
            return self._retriever

    def _load_retriever(self) -> SearchBackend:
        """Load the index (called on the first cache miss)."""
        return retriever_factory.create(self.backend, self.mode,
                                        self.index_dir)

    def _index_cache(self) -> Optional[CacheRepository]:
        """The cache, if enabled and an index exists to key it on."""
        if self._cache is None \
                or meta_repository.build_id(self.index_dir) is None:
            return None
        return self._cache

    def search(self, query: Any, k: Any) -> MinimalSearchResults:
        """Validate the arguments and run one search."""
        text = validator.query(query)
        k = validator.k(k)
        return MinimalSearchResults(
            question_id="query", question=text,
            retrieved_sources=self.retriever().search(text, k))

    def search_dataset(self, dataset_path: Any, k: Any,
                       save_directory: Any) -> str:
        """Search every question of a dataset; return the output path."""
        k = validator.k(k)
        dataset = self._load_dataset(dataset_path)
        json_repository.prepare_directory(save_directory)
        retriever = self.retriever()
        results = [
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=retriever.search(question.question, k))
            for question in tqdm(dataset.rag_questions, desc="Searching",
                                 unit="question")]
        output = StudentSearchResults(search_results=results, k=k)
        return json_repository.save(save_directory,
                                    os.path.basename(dataset_path),
                                    output.model_dump_json(indent=2))

    # ---- generation ----------------------------------------------------
    def answer(self, query: Any, k: Any,
               model_name: Any = settings.llm_model) -> MinimalAnswer:
        """Retrieve, then answer one question."""
        model_name = validator.text(model_name, "model_name")
        search = self.search(query, k)
        text = self._answer_text(search.question, search.retrieved_sources,
                                 model_name)
        return MinimalAnswer(**search.model_dump(), answer=text)

    def answer_dataset(self, search_results_path: Any, save_directory: Any,
                       model_name: Any = settings.llm_model) -> str:
        """Answer a whole ``StudentSearchResults`` file; return its path."""
        model_name = validator.text(model_name, "model_name")
        search = self._load_search_results(search_results_path)
        json_repository.prepare_directory(save_directory)
        total = len(search.search_results)
        print(f"Loaded {total} questions", file=sys.stderr)
        answers = [
            MinimalAnswer(**item.model_dump(),
                          answer=self._answer_text(item.question,
                                                   item.retrieved_sources,
                                                   model_name))
            for item in tqdm(search.search_results, desc="Answering",
                             unit="question")]
        print(f"Processed {len(answers)} of {total} questions",
              file=sys.stderr)
        output = StudentSearchResultsAndAnswer(search_results=answers,
                                               k=search.k)
        return json_repository.save(
            save_directory, os.path.basename(search_results_path),
            output.model_dump_json(indent=2))

    def _answer_text(self, question: str, sources: List[MinimalSource],
                     model_name: str) -> str:
        """Generate (or fetch from the cache) the answer to a question."""
        if not sources:
            return NO_CONTEXT_ANSWER
        cache = self._index_cache()
        key = CacheRepository.make_key(
            "answer", model_name, SYSTEM_PROMPT,
            meta_repository.build_id(self.index_dir), question,
            [source.model_dump() for source in sources])
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                return hit
        generator = self._generator(model_name)
        with self._generate_lock:
            text: str = generator.answer(question, sources)
        if cache is not None:
            cache.set(key, text)
        return text

    def _generator(self, model_name: str) -> Any:
        """Load the answer generator once per model name."""
        with self._lock:
            if model_name not in self._generators:
                try:
                    self._generators[model_name] = \
                        generator_provider.create(model_name)
                except ImportError as err:
                    raise RagError(f"generation dependencies missing "
                                   f"({err}); run 'uv sync'") from err
                except (OSError, ValueError) as err:
                    raise RagError(f"cannot load model {model_name}: "
                                   f"{err}") from err
            return self._generators[model_name]

    # ---- evaluation ----------------------------------------------------
    def evaluate(self, search_results_path: Any,
                 dataset_path: Any) -> EvaluationReport:
        """Recall@1/3/5/10 of a search results file."""
        search = self._load_search_results(search_results_path)
        dataset = self._load_dataset(dataset_path)
        try:
            return evaluator.evaluate(search, dataset)
        except ValueError as err:
            raise RagError(str(err)) from err

    # ---- input files ---------------------------------------------------
    @staticmethod
    def _load_dataset(path: Any) -> RagDataset:
        """Load and validate a RagDataset JSON file."""
        data = json_repository.load(path)
        try:
            return RagDataset.model_validate(data)
        except ValidationError as err:
            raise RagError(f"invalid dataset {path}: "
                           f"{err.error_count()} validation errors") from err

    @staticmethod
    def _load_search_results(path: Any) -> StudentSearchResults:
        """Load and validate a StudentSearchResults JSON file."""
        data = json_repository.load(path)
        try:
            return StudentSearchResults.model_validate(data)
        except ValidationError as err:
            raise RagError(f"invalid search results: {err.error_count()} "
                           "validation errors") from err
