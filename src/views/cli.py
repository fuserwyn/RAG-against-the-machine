"""Command-line interface (Python Fire): ``python -m src <command>``.

Every command only parses its flags, calls :class:`RagService` and turns
the result into text; the work itself lives in the service layer.
"""

import sys
from typing import Any

from ..config import settings
from ..errors import RagError
from ..services.rag_service import RagService
from ..services.validator import validator


class RagCLI:
    """RAG over the vLLM codebase: index, search, answer, evaluate, serve."""

    def __init__(self, index_dir: str = settings.index_dir,
                 backend: str = "native", mode: str = "lexical",
                 cache: bool = True) -> None:
        """Global flags, shared by every command.

        Args:
            index_dir: Root index directory (``<index_dir>/<backend>``).
            backend: ``native`` (own chunking + BM25), ``langchain``
                (LangChain splitters + BM25Retriever) or ``qdrant``
                (native chunks + MiniLM vectors in a Qdrant collection).
            mode: ``lexical`` (BM25), ``semantic`` (embeddings) or
                ``hybrid`` (both, fused); native backend only.
            cache: Cache search results and answers (``--cache False``).
        """
        self._service = RagService(index_dir, backend, mode, cache)

    def index(self, max_chunk_size: int = settings.max_chunk_size,
              raw_dir: str = settings.raw_dir, semantic: bool = False,
              incremental: bool = False) -> str:
        """Ingest ``raw_dir`` and build the index under ``index_dir``.

        Args:
            max_chunk_size: Maximum chunk length in characters (<= 2000).
            raw_dir: Corpus directory.
            semantic: Also embed every chunk (native backend only).
            incremental: Only re-index files changed since the last build
                (native backend only).
        """
        return self._service.index(max_chunk_size, raw_dir, semantic,
                                   incremental)

    def search(self, query: str, k: int = 10) -> str:
        """Return the top-k sources for a single query as JSON."""
        return self._service.search(query, k).model_dump_json(indent=2)

    def search_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = settings.search_output_dir
                       ) -> str:
        """Search every question of a dataset and save the results.

        Args:
            dataset_path: RagDataset JSON (answered or unanswered).
            k: Number of sources per question.
            save_directory: Output directory (file keeps the dataset name).
        """
        path = self._service.search_dataset(dataset_path, k, save_directory)
        return f"Saved student_search_results to {path}"

    def answer(self, query: str, k: int = 10,
               model_name: str = settings.llm_model) -> str:
        """Answer one question from the top-k retrieved sources."""
        result = self._service.answer(query, k, model_name)
        return result.model_dump_json(indent=2)

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str = settings.answer_output_dir,
                       model_name: str = settings.llm_model) -> str:
        """Generate answers for a StudentSearchResults file.

        Args:
            student_search_results_path: Output of ``search_dataset``.
            save_directory: Output directory.
            model_name: Hugging Face model id.
        """
        path = self._service.answer_dataset(student_search_results_path,
                                            save_directory, model_name)
        return f"Saved student_search_results_and_answer to {path}"

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> str:
        """Report recall@1/3/5/10 against a ground-truth dataset."""
        return str(self._service.evaluate(student_search_results_path,
                                          dataset_path))

    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> str:
        """Expose search and answer over a local HTTP API (JSON).

        Endpoints: ``GET /health``, ``GET|POST /search`` and
        ``GET|POST /answer`` (``query`` and ``k`` as URL parameters or
        JSON body). Stop with Ctrl+C.

        Args:
            host: Interface to bind (local only by default).
            port: TCP port (0 picks a free one).
        """
        from ..bonus.http_api.api import api_server_factory

        port = validator.port(port)
        host = validator.text(host, "host")
        self._service.retriever().load()
        try:
            server = api_server_factory.create(self._service, host, port)
        except OSError as err:
            raise RagError(f"cannot listen on {host}:{port}: {err}") from err
        with server:
            bound = server.server_address[1]
            print(f"Serving RAG API on http://{host}:{bound} "
                  "(Ctrl+C to stop)", file=sys.stderr)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        return "Server stopped."


def main() -> None:
    """Entry point: run Fire and turn every error into a clean message."""
    import fire

    result: Any = None
    try:
        result = fire.Fire(RagCLI)
    except RagError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as err:  # noqa: BLE001 - last-resort safety net
        print(f"Unexpected error: {type(err).__name__}: {err}",
              file=sys.stderr)
        sys.exit(1)
    del result
