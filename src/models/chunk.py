"""A chunk: one indexed piece of a corpus file."""

from typing import Tuple

from pydantic import BaseModel

from ..schemas.source import MinimalSource

Span = Tuple[int, int]          # [start, end) character offsets


class Chunk(BaseModel):
    """One indexed piece of a file.

    ``context`` holds extra searchable words (enclosing class or Markdown
    headings) that are indexed but are not part of the span itself.
    """

    file_path: str
    first_character_index: int
    last_character_index: int
    kind: str                   # "python" | "markdown" | "text"
    context: str = ""

    def body(self, text: str) -> str:
        """The part of the file ``text`` this chunk covers."""
        return text[self.first_character_index:self.last_character_index]

    def to_source(self) -> MinimalSource:
        """Convert the chunk into the span format expected by the grader."""
        return MinimalSource(
            file_path=self.file_path,
            first_character_index=self.first_character_index,
            last_character_index=self.last_character_index,
        )
