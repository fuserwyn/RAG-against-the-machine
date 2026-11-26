"""Extra: the MCP server that exposes the BM25 search as tools."""

import asyncio
import os
import sys
from typing import Any, Dict, List, Tuple

import pytest

from src.errors import RagError
from src.extra.mcp_server.tools import RagTools
from src.models.index_build import IndexReport

from .helpers import MiniCorpus

LORA_MD = "data/raw/proj/docs/features/lora.md"


def test_search_returns_ranked_hits_with_their_text(
        mini_corpus: MiniCorpus, mini_index: IndexReport) -> None:
    """Every hit carries the exact text of its span, best first."""
    tools = RagTools()
    hits = tools.search("lora adapter", 3)
    assert hits and [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert hits[0].file_path == LORA_MD
    for hit in hits:
        with open(hit.file_path, encoding="utf-8", newline="") as handle:
            text = handle.read()
        assert hit.text == text[hit.first_character_index:
                                hit.last_character_index]


@pytest.mark.parametrize("k", [0, -1, 21, True, "5"])
def test_search_rejects_a_bad_k(mini_corpus: MiniCorpus,
                                mini_index: IndexReport, k: Any) -> None:
    """k must be an integer in 1..20."""
    with pytest.raises(RagError, match="k must be"):
        RagTools().search("lora", k)


def test_search_rejects_an_empty_query(mini_corpus: MiniCorpus,
                                       mini_index: IndexReport) -> None:
    """The service validation applies to the tool too."""
    with pytest.raises(RagError):
        RagTools().search("   ", 3)


def test_read_source_returns_a_bounded_span(mini_corpus: MiniCorpus) -> None:
    """Characters [start, end) of a corpus file, capped in length."""
    tools = RagTools()
    tools.max_read_chars = 10
    part = tools.read_source(LORA_MD, 2, 500)
    full = mini_corpus.files[LORA_MD]
    assert (part.start, part.end) == (2, 12)
    assert part.text == full[2:12] and part.file_length == len(full)
    assert tools.read_source(LORA_MD, 0, None).end == 10


@pytest.mark.parametrize("path", ["../../etc/passwd", "/etc/passwd",
                                  "data/raw/../pyproject.toml",
                                  "data/raw/proj/missing.py",
                                  "data/raw/proj"])
def test_read_source_stays_inside_the_corpus(mini_corpus: MiniCorpus,
                                             path: str) -> None:
    """No path escapes data/raw, and only files can be read."""
    with pytest.raises(RagError, match="not a file of the corpus"):
        RagTools().read_source(path)


def test_read_source_rejects_a_reversed_range(
        mini_corpus: MiniCorpus) -> None:
    """start must not be after end."""
    with pytest.raises(RagError, match="start <= end"):
        RagTools().read_source(LORA_MD, 10, 5)


ServerReply = Tuple[List[str], Dict[str, Any], Any]


async def _talk_to_server(root: str) -> ServerReply:
    """Start the server as a subprocess and use it like an MCP client."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "src.extra.mcp_server"],
        cwd=root, env={**os.environ, "PYTHONPATH": project})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = [tool.name for tool in (await session.list_tools()).tools]
            found = await session.call_tool(
                "search_vllm", {"query": "lora adapter", "k": 2})
            denied = await session.call_tool(
                "read_source", {"file_path": "../../etc/passwd"})
    return names, found.structured_content or {}, denied


def test_server_speaks_mcp_over_stdio(mini_corpus: MiniCorpus,
                                      mini_index: IndexReport) -> None:
    """A real MCP client lists the tools, searches, gets clear errors."""
    names, found, denied = asyncio.run(
        _talk_to_server(str(mini_corpus.root)))
    assert sorted(names) == ["read_source", "search_vllm"]
    hits = found["result"]
    assert 1 <= len(hits) <= 2 and hits[0]["file_path"] == LORA_MD
    assert hits[0]["text"]
    assert denied.is_error
    assert "not a file of the corpus" in denied.content[0].text
