"""Low-level span arithmetic used by every chunker.

A span is a ``(start, end)`` pair of character offsets, end excluded.
"""

from typing import List

from ..models.chunk import Span


class SpanSplitter:
    """Cuts and merges character spans of a text."""

    def line_offsets(self, text: str) -> List[int]:
        """Character offset at which each line starts.

        The list has one extra final entry equal to ``len(text)`` so that
        line ``i`` always spans ``offsets[i]:offsets[i + 1]``.
        """
        offsets = [0]
        for line in text.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        if offsets[-1] != len(text):
            offsets.append(len(text))
        return offsets

    def split_by_lines(self, text: str, start: int, end: int,
                       max_size: int) -> List[Span]:
        """Split ``text[start:end]`` on line boundaries into spans <= max_size.

        A single line longer than ``max_size`` is cut into hard slices.
        """
        spans: List[Span] = []
        cur = start
        pos = start
        while pos < end:
            nl = text.find("\n", pos, end)
            line_end = end if nl == -1 else nl + 1
            if line_end - cur > max_size:
                if pos > cur:
                    spans.append((cur, pos))
                    cur = pos
                while line_end - cur > max_size:
                    spans.append((cur, cur + max_size))
                    cur += max_size
            pos = line_end
        if cur < end:
            spans.append((cur, end))
        return spans

    def merge_small(self, spans: List[Span], target: int) -> List[Span]:
        """Greedily merge adjacent contiguous spans while they fit."""
        merged: List[Span] = []
        for start, end in spans:
            if merged and merged[-1][1] == start \
                    and end - merged[-1][0] <= target:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged


span_splitter = SpanSplitter()
