"""Models: the internal domain objects of the pipeline.

Unlike ``schemas`` they never leave the program as JSON for the grader:
chunks, the BM25 matrix, build settings and reports.
"""

from .bm25_index import BM25Index
from .chunk import Chunk, Span
from .evaluation_report import EvaluationReport
from .index_build import BuildSettings, IndexReport

__all__ = ["BM25Index", "BuildSettings", "Chunk", "EvaluationReport",
           "IndexReport", "Span"]
