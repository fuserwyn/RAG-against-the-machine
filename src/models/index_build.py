"""What an indexing run takes as input and reports."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BuildSettings(BaseModel):
    """Parameters of a build: an index is reusable only if they match."""

    raw_dir: str
    max_chunk_size: int
    merge_ratio: float = 0.5

    @property
    def merge_target(self) -> int:
        """Small neighbouring pieces are merged up to this many chars."""
        return max(1, int(self.max_chunk_size * self.merge_ratio))

    def matches(self, meta: Dict[str, Any]) -> bool:
        """Whether a previous build (its ``meta.json``) used these values."""
        return all(meta.get(key) == value
                   for key, value in self.model_dump().items())


class IndexReport(BaseModel):
    """Summary of an indexing run.

    ``row_origin[i]`` is the row the i-th chunk had in the previous index
    when it was reused unchanged, or ``-1`` when it was (re)computed.
    """

    n_files: int
    n_chunks: int
    build_id: str
    incremental: bool = False
    previous_build_id: Optional[str] = None
    indexed_files: List[str] = Field(default_factory=list)
    reused_files: List[str] = Field(default_factory=list)
    removed_files: List[str] = Field(default_factory=list)
    row_origin: List[int] = Field(default_factory=list, repr=False)

    def summary(self) -> str:
        """One line describing what was (re)indexed."""
        if not self.incremental:
            return f"{self.n_chunks} chunks indexed"
        return (f"incremental update: {len(self.indexed_files)} file(s) "
                f"re-indexed, {len(self.reused_files)} unchanged, "
                f"{len(self.removed_files)} removed; {self.n_chunks} chunks")
