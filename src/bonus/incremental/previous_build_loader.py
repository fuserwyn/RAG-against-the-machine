"""Bonus 3: find the previous build an incremental run can reuse."""

from typing import Optional

from ...models.index_build import BuildSettings
from ...repositories.bm25_repository import BM25Repository, bm25_repository
from ...repositories.chunk_repository import ChunkRepository, chunk_repository
from ...repositories.meta_repository import MetaRepository, meta_repository
from .manifest_repository import ManifestRepository, manifest_repository
from .previous_build import PreviousBuild


class PreviousBuildLoader:
    """Loads the last build when its settings still match."""

    def __init__(self, meta: MetaRepository,
                 manifest: ManifestRepository,
                 chunks: ChunkRepository,
                 bm25: BM25Repository) -> None:
        """Read the index through these repositories."""
        self.meta = meta
        self.manifest = manifest
        self.chunks = chunks
        self.bm25 = bm25

    def load(self, index_dir: str,
             config: BuildSettings) -> Optional[PreviousBuild]:
        """The previous build, or ``None`` if it cannot be reused."""
        meta = self.meta.read(index_dir)
        build_id = meta.get("build_id")
        if not isinstance(build_id, str) or not config.matches(meta):
            return None
        try:
            digests = self.manifest.load(index_dir)
            chunks = self.chunks.load(index_dir)
            bm25 = self.bm25.load(index_dir, with_counts=True)
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if bm25.counts.shape[0] != len(chunks):
            return None
        return PreviousBuild(build_id=build_id, chunks=chunks,
                             digests=digests, bm25=bm25)


previous_build_loader = PreviousBuildLoader(
    meta=meta_repository, manifest=manifest_repository,
    chunks=chunk_repository, bm25=bm25_repository)
