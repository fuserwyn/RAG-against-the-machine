"""Record a real MCP session with the project's server, message by message.

The visualizer plays the role of the MCP client:
it starts ``python -m src.extra.mcp_server`` as a child process and writes
raw JSON-RPC lines to its stdin, reading the replies on its stdout. Every
line that crosses the pipe is recorded with its direction and time, so
the page can replay the exact exchange.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

PROTOCOL_VERSION = "2025-06-18"
REPLY_TIMEOUT = 120.0           # first search loads the index
MAX_STRING = 700                # long strings are clipped for display


def clip(value: Any) -> Any:
    """The same JSON with long strings shortened (display only)."""
    if isinstance(value, str) and len(value) > MAX_STRING:
        return (value[:MAX_STRING]
                + f" … [+{len(value) - MAX_STRING} chars clipped]")
    if isinstance(value, list):
        return [clip(item) for item in value]
    if isinstance(value, dict):
        return {key: clip(item) for key, item in value.items()}
    return value


class McpRecorder:
    """One stdio session with the server, recorded."""

    def __init__(self, cwd: str) -> None:
        """Start the server; the project root must be on PYTHONPATH."""
        self.t0 = time.perf_counter()
        self.messages: List[Dict[str, Any]] = []
        self.command = [sys.executable, "-m", "src.extra.mcp_server"]
        self.proc = subprocess.Popen(
            self.command, cwd=cwd, env=dict(os.environ),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            bufsize=1)
        self.lines: "queue.Queue[Optional[str]]" = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.next_id = 1

    def _pump(self) -> None:
        """Move every stdout line into the queue (None at EOF)."""
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def _ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 1)

    def _send(self, message: Dict[str, Any], note: str) -> None:
        assert self.proc.stdin is not None
        line = json.dumps(message)
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        self.messages.append({"dir": "out", "t": self._ms(), "note": note,
                              "bytes": len(line) + 1, "json": clip(message)})

    def notify(self, method: str, note: str) -> None:
        """A notification: no id, no reply."""
        self._send({"jsonrpc": "2.0", "method": method}, note)

    def request(self, method: str, params: Dict[str, Any],
                note: str) -> Dict[str, Any]:
        """Send a request and wait for the reply with the same id."""
        rid = self.next_id
        self.next_id += 1
        self._send({"jsonrpc": "2.0", "id": rid, "method": method,
                    "params": params}, note)
        deadline = time.monotonic() + REPLY_TIMEOUT
        while True:
            try:
                line = self.lines.get(
                    timeout=max(0.1, deadline - time.monotonic()))
            except queue.Empty as err:
                raise RuntimeError(f"no reply to {method}") from err
            if line is None:
                raise RuntimeError(f"server exited during {method}")
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                continue            # not protocol (should never happen)
            if not isinstance(reply, dict):
                continue
            same = reply.get("id") == rid
            self.messages.append({
                "dir": "in", "t": self._ms(), "bytes": len(line),
                "note": "" if same else "server message",
                "json": clip(reply), "reply_to": rid if same else None})
            if same:
                found: Dict[str, Any] = reply
                return found

    def close(self) -> None:
        """Close stdin: the server sees EOF and exits."""
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
        self.messages.append({"dir": "end", "t": self._ms(),
                              "note": f"stdin closed, server exited "
                                      f"(code {self.proc.returncode})"})


def record_session(query: str, k: int, cwd: str) -> Dict[str, Any]:
    """What an agent does: handshake, list tools, search, read, a mistake.

    The visualizer plays the model's decisions with fixed rules (search the
    question, read past the end of the first hit, try a path outside the
    corpus); a real client lets its model choose them.
    """
    rec = McpRecorder(cwd)
    try:
        init = rec.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "rag-visualizer", "version": "1.0"}},
            "handshake: protocol version, who we are")
        rec.notify("notifications/initialized", "ready")
        tools = rec.request("tools/list", {},
                            "which tools exist? (schemas go to the model)")
        found = rec.request("tools/call", {
            "name": "search_vllm",
            "arguments": {"query": query, "k": k}},
            "the model decides to search")
        hits = (found.get("result", {}).get("structuredContent") or {}) \
            .get("result", [])
        if hits:
            first = hits[0]
            end = first["last_character_index"]
            args = {"file_path": first["file_path"],
                    "start": end, "end": end + 600}
            rec.request("tools/call",
                        {"name": "read_source", "arguments": args},
                        "the model reads what follows hit #1")
        rec.request("tools/call", {
            "name": "read_source",
            "arguments": {"file_path": "../../etc/passwd"}},
            "a bad call: path outside the corpus")
    finally:
        rec.close()
    server = init.get("result", {})
    return {"command": " ".join(["python", "-m", "src.extra.mcp_server"]),
            "server": server.get("serverInfo", {}),
            "protocol": server.get("protocolVersion"),
            "tools": [t.get("name") for t in
                      tools.get("result", {}).get("tools", [])],
            "hits": len(hits), "messages": rec.messages}
