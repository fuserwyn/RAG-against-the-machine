"""Read access to the raw corpus (``data/raw``)."""

import hashlib
import os
from typing import FrozenSet, List, Tuple

from ..schemas.source import MinimalSource


class CorpusRepository:
    """Finds the indexable files and reads them exactly as the grader."""

    extensions: Tuple[str, ...] = (".py", ".md", ".rst", ".txt")
    skipped_dirs: FrozenSet[str] = frozenset(
        {".git", ".github", "__pycache__", "node_modules"})

    def discover(self, raw_dir: str) -> List[str]:
        """List the corpus files to index, as paths starting with raw_dir."""
        found: List[str] = []
        for root, dirs, files in os.walk(raw_dir):
            dirs[:] = sorted(d for d in dirs if d not in self.skipped_dirs)
            for name in sorted(files):
                if name.endswith(self.extensions):
                    path = os.path.normpath(os.path.join(root, name))
                    found.append(path.replace(os.sep, "/"))
        return found

    def read_text(self, path: str) -> str:
        """Read a file with no newline translation (character offsets)."""
        with open(path, encoding="utf-8", errors="replace",
                  newline="") as handle:
            return handle.read()

    def read_with_digest(self, path: str) -> Tuple[str, str]:
        """Read a file and return ``(text, sha1 of its bytes)``.

        The text is identical to :meth:`read_text`; the digest lets the
        incremental build detect modified files.
        """
        with open(path, "rb") as handle:
            data = handle.read()
        return (data.decode("utf-8", errors="replace"),
                hashlib.sha1(data).hexdigest())

    def read_span(self, source: MinimalSource) -> str:
        """Read the text covered by a source span."""
        text = self.read_text(source.file_path)
        return text[source.first_character_index:source.last_character_index]


corpus_repository = CorpusRepository()
