"""The SHA-1 of every indexed file (``manifest.json``).

The incremental build compares these digests with the files on disk to
find what was added, modified or removed since the last build.
"""

import json
import os
from typing import Dict


class ManifestRepository:
    """Reads and writes ``<index_dir>/manifest.json``."""

    file_name = "manifest.json"

    def save(self, index_dir: str, digests: Dict[str, str]) -> None:
        """Write the ``file path -> sha1`` map."""
        os.makedirs(index_dir, exist_ok=True)
        with open(os.path.join(index_dir, self.file_name), "w",
                  encoding="utf-8") as handle:
            json.dump({"files": digests}, handle)

    def load(self, index_dir: str) -> Dict[str, str]:
        """Read the ``file path -> sha1`` map.

        Raises:
            OSError: If the manifest is missing.
            ValueError: If it is not a JSON object of digests.
        """
        with open(os.path.join(index_dir, self.file_name),
                  encoding="utf-8") as handle:
            digests = json.load(handle).get("files")
        if not isinstance(digests, dict):
            raise ValueError("manifest.json has no 'files' object")
        return {str(path): str(digest) for path, digest in digests.items()}


manifest_repository = ManifestRepository()
