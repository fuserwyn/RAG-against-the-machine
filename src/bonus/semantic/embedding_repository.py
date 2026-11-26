"""Chunk vectors of the semantic bonus (``embeddings.npy``).

Row ``i`` is the vector of chunk ``i`` of ``chunks.jsonl``. A small
``semantic_meta.json`` records the encoder and the ``build_id`` of the
lexical index the vectors belong to.
"""

import json
import os
from typing import Any, Dict

import numpy as np


class EmbeddingRepository:
    """Reads and writes the vectors and their metadata."""

    vectors_file = "embeddings.npy"
    meta_file = "semantic_meta.json"

    def exists(self, index_dir: str) -> bool:
        """Whether embeddings were built in ``index_dir``."""
        return os.path.isfile(os.path.join(index_dir, self.vectors_file))

    def remove(self, index_dir: str) -> None:
        """Delete vectors that no longer match a rebuilt lexical index."""
        for name in (self.vectors_file, self.meta_file):
            path = os.path.join(index_dir, name)
            if os.path.isfile(path):
                os.remove(path)

    def read_meta(self, index_dir: str) -> Dict[str, Any]:
        """Metadata of the vectors, or ``{}`` if missing."""
        try:
            with open(os.path.join(index_dir, self.meta_file),
                      encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def load_vectors(self, index_dir: str) -> np.ndarray:
        """The ``(n_chunks, dim)`` float32 matrix.

        Raises:
            OSError, ValueError: If the file is missing or unreadable.
        """
        vectors = np.load(os.path.join(index_dir, self.vectors_file))
        return np.asarray(vectors, dtype=np.float32)

    def save(self, index_dir: str, vectors: np.ndarray,
             **meta: Any) -> None:
        """Write the vectors and their metadata."""
        np.save(os.path.join(index_dir, self.vectors_file), vectors)
        with open(os.path.join(index_dir, self.meta_file), "w",
                  encoding="utf-8") as handle:
            json.dump(meta, handle)


embedding_repository = EmbeddingRepository()
