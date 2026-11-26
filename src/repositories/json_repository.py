"""JSON files given on the command line (datasets, search results)."""

import json
import os
from typing import Any

from ..errors import RagError


class JsonRepository:
    """Reads input JSON and writes output JSON, with clear errors."""

    def load(self, path: Any) -> Any:
        """Read a JSON file, turning I/O and syntax errors into RagError."""
        if not isinstance(path, str) or not path:
            raise RagError("a file path is required")
        if not os.path.isfile(path):
            raise RagError(f"file not found: {path}")
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError) as err:
            raise RagError(f"malformed JSON in {path}: {err}") from err
        except OSError as err:
            raise RagError(f"cannot read {path}: {err}") from err

    def prepare_directory(self, directory: Any) -> str:
        """Create an output directory up front (fail before long work)."""
        if not isinstance(directory, str) or not directory:
            raise RagError("a save directory is required")
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as err:
            raise RagError(f"cannot write into {directory}: {err}") from err
        return directory

    def save(self, directory: str, name: str, payload: str) -> str:
        """Write ``payload`` to ``directory/name`` and return the path."""
        self.prepare_directory(directory)
        path = os.path.join(directory, name)
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(payload)
        except OSError as err:
            raise RagError(f"cannot write into {directory}: {err}") from err
        return path


json_repository = JsonRepository()
