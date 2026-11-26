"""IX - Bonus 5: local HTTP API."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pytest

from src.models.index_build import IndexReport
from src.schemas import MinimalAnswer, MinimalSearchResults
from src.bonus.http_api.api import api_server_factory
from src.services.rag_service import RagService

from .helpers import CLIResult, GenerateCalls, MiniCorpus, fake_answer

LORA_MD = "data/raw/proj/docs/features/lora.md"


@pytest.fixture
def api(mini_corpus: MiniCorpus, mini_index: IndexReport,
        fake_generator: GenerateCalls) -> Iterator[str]:
    """Start the API on a free port and return its base URL."""
    server = api_server_factory.create(RagService(), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def call(url: str, body: Optional[bytes] = None,
         method: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
    """Send a request and return ``(status, decoded JSON)``."""
    request = Request(url, data=body, method=method,
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except HTTPError as err:
        with err:
            return err.code, json.loads(err.read())


def post(url: str, payload: Any) -> Tuple[int, Dict[str, Any]]:
    """POST a JSON payload."""
    return call(url, json.dumps(payload).encode(), "POST")


def test_health(api: str) -> None:
    """GET /health reports the backend and mode."""
    status, body = call(f"{api}/health")
    assert status == 200
    assert body["status"] == "ok" and body["backend"] == "native"
    assert body["mode"] == "lexical"


def test_search_with_get_and_post(api: str) -> None:
    """/search answers GET parameters and POST JSON alike."""
    status, by_get = call(f"{api}/search?query={quote('load lora')}&k=2")
    assert status == 200
    parsed = MinimalSearchResults.model_validate(by_get)
    assert parsed.retrieved_sources[0].file_path == LORA_MD
    assert len(parsed.retrieved_sources) <= 2
    status, by_post = post(f"{api}/search", {"query": "load lora", "k": 2})
    assert status == 200 and by_post == by_get


def test_answer_endpoint(api: str, fake_generator: GenerateCalls) -> None:
    """/answer returns a MinimalAnswer from the retrieved context."""
    status, body = post(f"{api}/answer", {"query": "load lora", "k": 2})
    assert status == 200
    answer = MinimalAnswer.model_validate(body)
    assert answer.answer == fake_answer("load lora",
                                        answer.retrieved_sources)
    status, again = call(f"{api}/answer?query=load%20lora&k=2")
    assert status == 200 and again == body
    assert fake_generator.loads == 1


@pytest.mark.parametrize("path,payload,message", [
    ("/search", {"query": ""}, "the query is empty"),
    ("/search", {"query": "x", "k": 0}, "k must be positive"),
    ("/search", {"query": "x", "k": "three"}, "k must be an integer"),
    ("/search", {"query": "x", "k": True}, "k must be an integer"),
    ("/search", [1, 2], "must be an object"),
    ("/answer", {"query": "  "}, "the query is empty"),
    ("/answer", {"query": "x", "model_name": 3}, "model_name"),
])
def test_invalid_requests_get_400(api: str, path: str, payload: Any,
                                  message: str) -> None:
    """Bad input is a 400 with an error message, never a crash."""
    status, body = post(f"{api}{path}", payload)
    assert status == 400 and message in body["error"]


def test_malformed_json_body(api: str) -> None:
    """A body that is not JSON is a 400."""
    status, body = call(f"{api}/search", b"{nope", "POST")
    assert status == 400 and "malformed JSON" in body["error"]


def test_bad_k_in_url(api: str) -> None:
    """A non-integer k in the URL is a 400."""
    status, body = call(f"{api}/search?query=lora&k=ten")
    assert status == 400 and "k must be an integer" in body["error"]


def test_unknown_endpoint_and_method(api: str) -> None:
    """Unknown paths are 404, wrong methods 405."""
    assert call(f"{api}/nope")[0] == 404
    assert post(f"{api}/health", {})[0] == 405


def test_concurrent_requests(api: str) -> None:
    """The threaded server answers parallel clients consistently."""
    def ask(i: int) -> Tuple[int, Dict[str, Any]]:
        return call(f"{api}/search?query=scheduler%20method%20{i % 3}&k=3")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(ask, range(24)))
    assert all(status == 200 for status, _ in results)
    assert results[0][1] == results[3][1]


def test_server_survives_errors(api: str) -> None:
    """After bad requests the server still serves good ones."""
    post(f"{api}/search", {"query": ""})
    call(f"{api}/search", b"{nope", "POST")
    assert call(f"{api}/health")[0] == 200


@pytest.mark.parametrize("args,message", [
    (["serve", "--port", "70000"], "port must be an integer"),
    (["serve", "--port", "http"], "port must be an integer"),
])
def test_serve_command_validates_its_options(
        mini_corpus: MiniCorpus, mini_index: IndexReport,
        cli: Callable[..., CLIResult], args: Any, message: str) -> None:
    """``serve`` refuses invalid ports before binding."""
    result = cli(*args)
    assert result.code == 1 and message in result.err


def test_serve_command_needs_an_index(mini_corpus: MiniCorpus,
                                      cli: Callable[..., CLIResult]) -> None:
    """``serve`` fails fast when there is nothing to query."""
    result = cli("serve", "--port", "0")
    assert result.code == 1 and "run the 'index' command" in result.err
