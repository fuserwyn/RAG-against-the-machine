"""The chunk list of the native index (``chunks.jsonl``).

One JSON object per line; the line number is the chunk id, which is also
its row in the BM25 matrix and in ``embeddings.npy``. Only the span is
stored, the text is read back from ``data/raw`` when needed.
"""

import os
from typing import List

from ..models.chunk import Chunk


class ChunkRepository:
    """Reads and writes ``<index_dir>/chunks.jsonl``."""

    file_name = "chunks.jsonl"

    def save(self, index_dir: str, chunks: List[Chunk]) -> None:
        """Write one JSON chunk per line."""
        os.makedirs(index_dir, exist_ok=True)
        with open(os.path.join(index_dir, self.file_name), "w",
                  encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(chunk.model_dump_json() + "\n")

    def load(self, index_dir: str) -> List[Chunk]:
        """Read the chunks of an existing index.

        Raises:
            FileNotFoundError: If the index has not been built yet.
        """
        path = os.path.join(index_dir, self.file_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"no index found in {index_dir!r}: run the 'index' command "
                "first")
        with open(path, encoding="utf-8") as handle:
            return [Chunk.model_validate_json(line)
                    for line in handle if line.strip()]


chunk_repository = ChunkRepository()
