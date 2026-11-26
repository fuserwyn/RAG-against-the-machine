# RAG Blocks

An isometric, block-by-block view of this project's RAG pipeline. It runs the project's real code (`src/` and the indexes in `data/processed/`), not a mock.

## Run

```bash
make up      # Docker: build and start rag-engine + rag-viz, then open http://localhost:8080
make down
make local   # no Docker: same server in the project's uv env, http://localhost:8080
```

Index the backends you want to show first:

```bash
cd ..   # repository root
uv run python -m src index
uv run python -m src --backend langchain index
uv run python -m src --backend qdrant index      # ~2 min (embeds every chunk)
```

## Architecture

```
browser ──:8080──▶ rag-viz (nginx) ──/api──▶ rag-engine:8000 (python, CPU torch)
                   └────────────── rag-net network ─────────────┘     │
                                    repository root (ro)   ~/.cache/huggingface
```

- `rag-engine` (`engine/trace_server.py`) imports the project's modules and runs a question stage by stage: tokenize (or embed, for Qdrant), retrieve (native BM25, LangChain BM25 or Qdrant vectors), context packing, Qwen3. For every stage it returns timings and data: each term's share of the BM25 score, the token budget, the full prompt and the final JSON.
- The MCP card (`POST /api/mcp`, `engine/mcp_trace.py`) plays an MCP client: it starts `python -m src.extra.mcp_server`, writes raw JSON-RPC lines to its stdin (initialize, tools/list, search_vllm, read_source, one bad call) and records every line with its direction, time and size; the page replays them as a sequence diagram.
- `rag-viz` is nginx with one static page. It proxies `/api` to `rag-engine` by name inside the network, and port 8000 is never published to the host.

The embedded Qdrant storage can be open in one process only: while the visualizer has used the Qdrant backend, stop it before running `--backend qdrant` from the CLI.

Run every command from `visualizer/`. The visualizer is a demo tool: the graded pipeline is the `src` CLI.
