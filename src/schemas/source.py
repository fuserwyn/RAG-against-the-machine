"""The source location format the grader expects."""

from typing import List

from pydantic import BaseModel, TypeAdapter


class MinimalSource(BaseModel):
    """A character span inside one file of the corpus."""

    file_path: str
    first_character_index: int
    last_character_index: int

    @property
    def length(self) -> int:
        """Number of characters covered by the span."""
        return self.last_character_index - self.first_character_index


# validates / serializes a whole list of sources at once (cache entries)
sources_adapter = TypeAdapter(List[MinimalSource])
