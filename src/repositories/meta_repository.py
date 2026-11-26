"""Index metadata (``meta.json``), written by every backend.

Every build writes a fresh random ``build_id``. Derived data (embeddings,
cached query results) records the ``build_id`` it was computed for, so it
can tell when the index under it has been rebuilt.
"""

import json
import os
import uuid
from typing import Any, Dict, Optional


class MetaRepository:
    """Reads and writes ``<index_dir>/meta.json``."""

    file_name = "meta.json"

    def write(self, index_dir: str, **fields: Any) -> str:
        """Write the metadata with a new build id and return that id."""
        build_id = uuid.uuid4().hex
        os.makedirs(index_dir, exist_ok=True)
        with open(os.path.join(index_dir, self.file_name), "w",
                  encoding="utf-8") as handle:
            json.dump({**fields, "build_id": build_id}, handle)
        return build_id

    def read(self, index_dir: str) -> Dict[str, Any]:
        """Return the metadata of an index, or ``{}`` if it is missing."""
        try:
            with open(os.path.join(index_dir, self.file_name),
                      encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def build_id(self, index_dir: str) -> Optional[str]:
        """Return the build id of an index, or ``None`` if there is none."""
        value = self.read(index_dir).get("build_id")
        return value if isinstance(value, str) and value else None


meta_repository = MetaRepository()
