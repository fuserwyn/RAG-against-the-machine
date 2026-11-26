"""Persistent key/value cache for search results and answers (bonus 4).

Entries live in a small SQLite database inside the index directory, with
an in-memory LRU in front of it. Keys contain the ``build_id`` of the
index, so a rebuild makes older entries unreachable (``index`` also
deletes the file).
"""

import hashlib
import json
import os
import sqlite3
import threading
from collections import OrderedDict
from contextlib import closing
from typing import Any, Optional


class CacheRepository:
    """SQLite on disk plus an in-memory LRU.

    The cache is best effort: if the database cannot be opened or written
    (missing directory, read-only disk, ...) it simply behaves as empty.
    """

    file_name = "query_cache.sqlite"

    def __init__(self, path: str, memory_size: int = 1024) -> None:
        """Create a cache stored in the SQLite file ``path``."""
        self.path = path
        self.memory_size = memory_size
        self._memory: "OrderedDict[str, str]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @classmethod
    def in_index(cls, index_dir: str) -> "CacheRepository":
        """The cache file that belongs to an index directory."""
        return cls(os.path.join(index_dir, cls.file_name))

    @staticmethod
    def make_key(*parts: Any) -> str:
        """Hash JSON-serializable parts into a fixed-size key."""
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        """Open the database, creating the table if needed."""
        conn = sqlite3.connect(self.path, timeout=10)
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("CREATE TABLE IF NOT EXISTS cache "
                     "(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        return conn

    def _remember(self, key: str, value: str) -> None:
        """Store an entry in the in-memory LRU."""
        with self._lock:
            self._memory[key] = value
            self._memory.move_to_end(key)
            while len(self._memory) > self.memory_size:
                self._memory.popitem(last=False)

    def get(self, key: str) -> Optional[str]:
        """Return the cached value for ``key``, or ``None``."""
        with self._lock:
            if key in self._memory:
                self._memory.move_to_end(key)
                self.hits += 1
                return self._memory[key]
        row = None
        if os.path.isfile(self.path):
            try:
                with closing(self._connect()) as conn:
                    row = conn.execute("SELECT value FROM cache WHERE key = ?",
                                       (key,)).fetchone()
            except sqlite3.Error:
                row = None
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        self._remember(key, str(row[0]))
        return str(row[0])

    def set(self, key: str, value: str) -> None:
        """Store ``value`` under ``key`` (in memory and on disk)."""
        self._remember(key, value)
        try:
            with closing(self._connect()) as conn, conn:
                conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)",
                             (key, value))
        except sqlite3.Error:
            pass

    def clear(self) -> None:
        """Forget every entry and delete the database file."""
        with self._lock:
            self._memory.clear()
        for suffix in ("", "-journal"):
            try:
                os.remove(self.path + suffix)
            except FileNotFoundError:
                continue
