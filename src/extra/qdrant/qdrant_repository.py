"""The Qdrant collection (embedded mode: a folder, no server).

Every chunk is one point: ``id`` = chunk number, ``vector`` = its MiniLM
embedding, ``payload`` = the source span plus kind and context.
"""

import os
import shutil
import warnings
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm


class QdrantRepository:
    """Creates, fills and opens ``<index_dir>/storage``."""

    collection = "chunks"
    storage_dir = "storage"
    upsert_batch = 512

    def storage(self, index_dir: str) -> str:
        """Folder of the embedded Qdrant database."""
        return os.path.join(index_dir, self.storage_dir)

    def save(self, index_dir: str, vectors: np.ndarray,
             payloads: List[Dict[str, Any]]) -> None:
        """Replace the collection by these points (cosine distance)."""
        from qdrant_client import QdrantClient, models

        storage = self.storage(index_dir)
        if os.path.isdir(storage):
            shutil.rmtree(storage)
        os.makedirs(index_dir, exist_ok=True)
        client = QdrantClient(path=storage)
        try:
            client.create_collection(
                self.collection, vectors_config=models.VectorParams(
                    size=int(vectors.shape[1]),
                    distance=models.Distance.COSINE))
            for start in tqdm(range(0, len(payloads), self.upsert_batch),
                              desc="Upserting into Qdrant", unit="batch"):
                end = start + self.upsert_batch
                client.upsert(self.collection, points=[
                    models.PointStruct(id=start + offset,
                                       vector=vectors[start + offset].tolist(),
                                       payload=payload)
                    for offset, payload in enumerate(payloads[start:end])])
        finally:
            client.close()

    def open(self, index_dir: str) -> Any:
        """Open the storage for searching.

        Raises:
            FileNotFoundError: If the index has not been built yet.
            RuntimeError: If another process already holds the storage.
        """
        from qdrant_client import QdrantClient

        storage = self.storage(index_dir)
        if not os.path.isdir(storage):
            raise FileNotFoundError(
                f"no qdrant index found in {index_dir!r}: run "
                "'--backend qdrant index' first")
        try:
            with warnings.catch_warnings():
                # embedded mode warns above 20k points; this corpus has
                # ~23k and an exact numpy search over them takes ~10 ms
                warnings.filterwarnings("ignore", message="Local mode")
                return QdrantClient(path=storage)
        except RuntimeError as err:
            raise RuntimeError(
                f"the Qdrant storage {storage!r} is already open in another "
                "process (embedded mode allows one): stop it and retry"
            ) from err

    def nearest(self, client: Any, vector: np.ndarray,
                k: int) -> List[Any]:
        """The ``k`` points closest to ``vector`` (with payloads)."""
        return list(client.query_points(self.collection,
                                        query=vector.tolist(), limit=k,
                                        with_payload=True).points)


qdrant_repository = QdrantRepository()
