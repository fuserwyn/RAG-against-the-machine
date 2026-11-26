"""What an incremental build reuses from the build before it."""

from dataclasses import dataclass
from typing import Dict, List

from ...models.bm25_index import BM25Index
from ...models.chunk import Chunk


@dataclass
class PreviousBuild:
    """What an incremental build reuses from the build before it."""

    build_id: str
    chunks: List[Chunk]
    digests: Dict[str, str]         # file path -> sha1 of its bytes
    bm25: BM25Index

    def rows_by_file(self) -> Dict[str, List[int]]:
        """Matrix rows of every file, in corpus order."""
        rows: Dict[str, List[int]] = {}
        for row, chunk in enumerate(self.chunks):
            rows.setdefault(chunk.file_path, []).append(row)
        return rows
