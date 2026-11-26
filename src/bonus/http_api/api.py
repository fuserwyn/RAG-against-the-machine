"""Local HTTP API (bonus 5): query the index and answer over HTTP.

Built on the standard library (``http.server``), JSON in and JSON out:

    GET  /health
    GET  /search?query=...&k=5      POST /search  {"query": "...", "k": 5}
    GET  /answer?query=...&k=5      POST /answer  {"query": "...", "k": 5}

Responses are the same pydantic models the CLI prints
(``MinimalSearchResults`` / ``MinimalAnswer``). Invalid requests get a
``400`` with ``{"error": "..."}``; the server never crashes on bad input.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Tuple, cast
from urllib.parse import parse_qs, urlparse

from ...config import settings
from ...errors import RagError
from ...services.rag_service import RagService


class RagHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server that holds the shared RAG service."""

    daemon_threads = True

    def __init__(self, address: Tuple[str, int], service: RagService) -> None:
        """Bind ``address`` and serve requests with ``service``."""
        super().__init__(address, RagRequestHandler)
        self.service = service


class RagRequestHandler(BaseHTTPRequestHandler):
    """Routes requests to the RAG service and writes JSON responses."""

    server_version = "RAGAgainstTheMachine/1.0"
    max_body_bytes = 1 << 20

    def do_GET(self) -> None:
        """Handle a GET request."""
        self._dispatch("GET")

    def do_POST(self) -> None:
        """Handle a POST request."""
        self._dispatch("POST")

    def log_message(self, format: str, *args: Any) -> None:
        """Log one line per request on stderr."""
        print(f"[api] {self.address_string()} {format % args}",
              file=sys.stderr)

    def _dispatch(self, method: str) -> None:
        """Run the endpoint and always answer with JSON."""
        service = cast(RagHTTPServer, self.server).service
        url = urlparse(self.path)
        try:
            status, payload = self._route(service, method, url.path,
                                          url.query)
        except RagError as err:
            status, payload = 400, {"error": str(err)}
        except Exception as err:  # noqa: BLE001 - keep the server alive
            status, payload = 500, {"error": f"{type(err).__name__}: {err}"}
        self._send(status, payload)

    def _route(self, service: RagService, method: str, path: str,
               query: str) -> Tuple[int, Dict[str, Any]]:
        """Return ``(status, body)`` for one request."""
        if path == "/health":
            if method != "GET":
                return 405, {"error": "use GET"}
            return 200, {"status": "ok", **service.info()}
        if path not in ("/search", "/answer"):
            return 404, {"error": f"unknown endpoint {path}"}
        params = (self._query_params(query) if method == "GET"
                  else self._json_body())
        text = params.get("query", "")
        k = params.get("k", 10)
        if path == "/search":
            return 200, service.search(text, k).model_dump()
        model = params.get("model_name", settings.llm_model)
        return 200, service.answer(text, k, model).model_dump()

    @staticmethod
    def _query_params(query: str) -> Dict[str, Any]:
        """Parse URL parameters; ``k`` is converted to an integer."""
        params: Dict[str, Any] = {key: values[-1] for key, values
                                  in parse_qs(query).items()}
        if "k" in params:
            try:
                params["k"] = int(params["k"])
            except ValueError as err:
                raise RagError(f"k must be an integer, got "
                               f"{params['k']!r}") from err
        return params

    def _json_body(self) -> Dict[str, Any]:
        """Read and decode a JSON object request body."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as err:
            raise RagError("invalid Content-Length header") from err
        if length < 0 or length > self.max_body_bytes:
            raise RagError(
                f"request body must be 0..{self.max_body_bytes} bytes")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            raise RagError(f"malformed JSON body: {err}") from err
        if not isinstance(body, dict):
            raise RagError("the JSON body must be an object")
        return body

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        """Write a JSON response."""
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ApiServerFactory:
    """Creates (but does not start) HTTP servers."""

    def create(self, service: RagService, host: str,
               port: int) -> RagHTTPServer:
        """Bind ``host:port``.

        Raises:
            OSError: If the address cannot be bound.
        """
        return RagHTTPServer((host, port), service)


api_server_factory = ApiServerFactory()
