"""Project-wide settings: every default path, limit and model name.

Each layer reads its defaults from the single ``settings`` object below,
so a value (the grader's span limit, the LLM name, ...) is written once.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Settings:
    """Default values shared by the whole pipeline."""

    raw_dir: str = "data/raw"
    index_dir: str = "data/processed"
    search_output_dir: str = "data/output/search_results"
    answer_output_dir: str = "data/output/search_results_and_answer"

    max_chunk_size: int = 2000
    max_context_length: int = 2000      # grader limit for one source
    merge_ratio: float = 0.5

    llm_model: str = "Qwen/Qwen3-0.6B"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_context_tokens: int = 3000
    max_new_tokens: int = 256

    backends: Tuple[str, ...] = ("native", "langchain", "qdrant")
    modes: Tuple[str, ...] = ("lexical", "semantic", "hybrid")
    recall_ks: Tuple[int, ...] = (1, 3, 5, 10)


settings = Settings()
