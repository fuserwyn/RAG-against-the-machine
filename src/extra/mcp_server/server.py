"""MCP server: the client's model calls our search as a tool (agentic RAG).

    MCP client
        │  tools/list  -> search_vllm, read_source (+ JSON schemas)
        │  tools/call  -> search_vllm(query="speculative_config", k=5)
        ▼
    this server (stdio, JSON-RPC) -> RagTools -> native BM25 index

Run it with ``python -m src.extra.mcp_server`` from the repository root.
Nothing may be printed on stdout: it is the protocol channel.
"""

import os
from typing import List, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ...config import settings
from ...errors import RagError
from .tools import RagTools, SearchHit, SourceText

INSTRUCTIONS = (
    "Search the vLLM 0.10.1 source code and documentation. Use "
    "search_vllm with the words the code is likely to contain "
    "(function, class, flag or config names), several times if needed, "
    "then read_source to see more around a hit. Cite file paths."
)


class RagMcpServerFactory:
    """Builds the MCP server around a :class:`RagTools`."""

    name = "vllm-rag"

    def create(self, tools: RagTools) -> MCPServer:
        """Register the tools; the server is not started."""
        server: MCPServer = MCPServer(name=self.name,
                                      instructions=INSTRUCTIONS)

        @server.tool()
        def search_vllm(query: str, k: int = 5) -> List[SearchHit]:
            """Find the places of the vLLM 0.10.1 code base about a query.

            Lexical BM25 search over 23k chunks (functions, classes,
            Markdown sections). Returns the k best chunks with their file
            path, character span and text. Use code words: identifiers,
            CLI flags, config keys.
            """
            try:
                return tools.search(query, k)
            except RagError as err:     # shown to the model, not a crash
                raise ToolError(str(err)) from err

        @server.tool()
        def read_source(file_path: str, start: int = 0,
                        end: Optional[int] = None) -> SourceText:
            """Read characters [start, end) of a file of the vLLM corpus.

            file_path is a path returned by search_vllm. At most 6000
            characters are returned; use it to see around a search hit.
            """
            try:
                return tools.read_source(file_path, start, end)
            except RagError as err:
                raise ToolError(str(err)) from err

        return server


rag_mcp_server_factory = RagMcpServerFactory()


def main() -> None:
    """Serve over stdio (``RAG_INDEX_DIR`` / ``RAG_RAW_DIR`` override)."""
    tools = RagTools(
        index_dir=os.environ.get("RAG_INDEX_DIR", settings.index_dir),
        raw_dir=os.environ.get("RAG_RAW_DIR", settings.raw_dir))
    rag_mcp_server_factory.create(tools).run("stdio")
