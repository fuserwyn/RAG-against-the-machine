"""The native chunking strategies: Python code, Markdown, plain text.

Every chunk is a contiguous character span ``[start, end)`` of the
original file and is never longer than ``max_size`` characters.
``FileChunker`` picks the strategy from the file extension.
"""

import ast
import re
from typing import List, Tuple

from ..models.chunk import Chunk, Span
from .span_splitter import SpanSplitter, span_splitter

Section = Tuple[int, int, str]          # (start, end, context)


class TextChunker:
    """Plain text (and unparsable code): cut on blank lines."""

    def __init__(self, splitter: SpanSplitter) -> None:
        """Use ``splitter`` for the span arithmetic."""
        self.splitter = splitter

    def split_paragraphs(self, text: str, start: int, end: int,
                         max_size: int) -> List[Span]:
        """Split a region on blank lines, then on lines if still too big."""
        pieces: List[Span] = []
        cur = start
        for match in re.finditer(r"\n[ \t]*\n", text[start:end]):
            cut = start + match.end()
            pieces.append((cur, cut))
            cur = cut
        if cur < end:
            pieces.append((cur, end))
        spans: List[Span] = []
        for lo, hi in pieces:
            if hi - lo > max_size:
                spans.extend(self.splitter.split_by_lines(text, lo, hi,
                                                          max_size))
            else:
                spans.append((lo, hi))
        return self.splitter.merge_small(spans, max_size)

    def chunk(self, text: str, file_path: str, max_size: int,
              merge_target: int) -> List[Chunk]:
        """Paragraphs, merged up to ``merge_target`` characters."""
        spans = self.split_paragraphs(text, 0, len(text), max_size)
        spans = self.splitter.merge_small(spans, merge_target)
        return [Chunk(file_path=file_path, first_character_index=lo,
                      last_character_index=hi, kind="text")
                for lo, hi in spans if text[lo:hi].strip()]


class PythonChunker:
    """Python code: cut along the syntax tree (``ast``)."""

    def __init__(self, splitter: SpanSplitter,
                 fallback: TextChunker) -> None:
        """Use ``fallback`` for files that do not parse."""
        self.splitter = splitter
        self.fallback = fallback

    def chunk(self, text: str, file_path: str, max_size: int,
              merge_target: int) -> List[Chunk]:
        """One chunk per top-level definition; big classes per method.

        Comments and decorators stay with the definition they precede, and
        small neighbouring pieces are merged up to ``merge_target``.
        """
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return self.fallback.chunk(text, file_path, max_size,
                                       merge_target)
        offsets = self.splitter.line_offsets(text)
        sections = self._spans(text, offsets, tree.body, 0, len(text),
                               max_size, "")
        chunks: List[Chunk] = []
        for start, end, context in sections:
            last = chunks[-1] if chunks else None
            if last is not None and last.last_character_index == start \
                    and last.context == context \
                    and end - last.first_character_index <= merge_target:
                last.last_character_index = end
                continue
            chunks.append(Chunk(file_path=file_path,
                                first_character_index=start,
                                last_character_index=end, kind="python",
                                context=context))
        return [c for c in chunks if c.body(text).strip()]

    def _node_start(self, node: ast.stmt, text: str,
                    offsets: List[int]) -> int:
        """Offset where a statement starts, decorators and comments included.

        Comment lines directly above the statement (no blank line between)
        belong to it, so they stay in the same chunk.
        """
        decorators: List[ast.expr] = getattr(node, "decorator_list", [])
        line: int = min([node.lineno] + [d.lineno for d in decorators])
        while line > 1 and text[offsets[line - 2]:offsets[line - 1]] \
                .lstrip().startswith("#"):
            line -= 1
        return offsets[line - 1]

    def _spans(self, text: str, offsets: List[int], body: List[ast.stmt],
               start: int, end: int, max_size: int,
               prefix: str) -> List[Section]:
        """Recursively cut a block of statements into sections.

        ``[start, end)`` is divided at the first line of every statement
        of ``body``. A class or function that is still too big is split
        again at its own statements; anything else falls back to lines.
        Each section carries the names of its enclosing classes.
        """
        bounds = [start]
        for node in body:
            pos = self._node_start(node, text, offsets)
            if start < pos < end:
                bounds.append(pos)
        bounds.append(end)
        result: List[Section] = []
        for lo, hi in zip(bounds, bounds[1:]):
            if hi <= lo:
                continue
            if hi - lo <= max_size:
                result.append((lo, hi, prefix))
                continue
            target = next((n for n in body
                           if self._node_start(n, text, offsets) == lo),
                          None)
            if isinstance(target, (ast.ClassDef, ast.FunctionDef,
                                   ast.AsyncFunctionDef)) and target.body:
                name = f"{prefix} {target.name}".strip()
                result.extend(self._spans(text, offsets, target.body, lo,
                                          hi, max_size, name))
            else:
                result.extend(
                    (a, b, prefix) for a, b in
                    self.splitter.split_by_lines(text, lo, hi, max_size))
        return result


class MarkdownChunker:
    """Markdown / reStructuredText: cut at headings, then paragraphs."""

    HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
    FENCE_RE = re.compile(r"^\s*(```|~~~)")

    def __init__(self, splitter: SpanSplitter,
                 paragraphs: TextChunker) -> None:
        """Use ``paragraphs`` to cut sections that are too long."""
        self.splitter = splitter
        self.paragraphs = paragraphs

    def chunk(self, text: str, file_path: str, max_size: int,
              merge_target: int) -> List[Chunk]:
        """One chunk per heading section (headings path as context).

        Small consecutive sections are merged up to ``merge_target``.
        """
        chunks: List[Chunk] = []
        for start, end, title in self.sections(text):
            if end - start <= max_size:
                spans = [(start, end)]
            else:
                spans = self.paragraphs.split_paragraphs(text, start, end,
                                                         max_size)
            for lo, hi in spans:
                last = chunks[-1] if chunks else None
                if last is not None and last.last_character_index == lo \
                        and hi - last.first_character_index <= merge_target:
                    last.last_character_index = hi
                    last.context += " " + title
                    continue
                chunks.append(Chunk(file_path=file_path,
                                    first_character_index=lo,
                                    last_character_index=hi,
                                    kind="markdown", context=title))
        return [c for c in chunks if c.body(text).strip()]

    def sections(self, text: str) -> List[Section]:
        """Sections starting at each heading, with their heading path.

        Headings inside fenced code blocks are ignored. The path looks
        like ``"LoRA > Dynamic serving"``.
        """
        offsets = self.splitter.line_offsets(text)
        starts: List[Tuple[int, str]] = [(0, "")]
        stack: List[Tuple[int, str]] = []
        in_fence = False
        for i, line in enumerate(text.splitlines(keepends=True)):
            if self.FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            match = None if in_fence else self.HEADING_RE.match(line)
            if match:
                level = len(match.group(1))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, match.group(2).strip()))
                starts.append((offsets[i], " > ".join(t for _, t in stack)))
        sections: List[Section] = []
        for idx, (pos, title) in enumerate(starts):
            end = starts[idx + 1][0] if idx + 1 < len(starts) else len(text)
            if end > pos:
                sections.append((pos, end, title))
        return sections


class FileChunker:
    """Sends each file to the strategy that matches its extension."""

    def __init__(self, python: PythonChunker, markdown: MarkdownChunker,
                 text: TextChunker) -> None:
        """Wire the three strategies."""
        self.python = python
        self.markdown = markdown
        self.text = text

    def chunk(self, text: str, file_path: str, max_size: int,
              merge_target: int) -> List[Chunk]:
        """Chunk one file (``.py`` / ``.md`` ``.rst`` / anything else)."""
        if file_path.endswith(".py"):
            strategy = self.python.chunk
        elif file_path.endswith((".md", ".rst")):
            strategy = self.markdown.chunk
        else:
            strategy = self.text.chunk
        return strategy(text, file_path, max_size, merge_target)


text_chunker = TextChunker(span_splitter)
python_chunker = PythonChunker(span_splitter, fallback=text_chunker)
markdown_chunker = MarkdownChunker(span_splitter, paragraphs=text_chunker)
file_chunker = FileChunker(python_chunker, markdown_chunker, text_chunker)
