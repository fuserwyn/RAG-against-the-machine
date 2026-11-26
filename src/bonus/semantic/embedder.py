"""Sentence embeddings with a small CPU model (all-MiniLM-L6-v2).

Every text becomes one L2-normalized 384-number vector, so the cosine
similarity of two texts is a plain dot product.
"""

from typing import Any, Callable, List

import numpy as np
from tqdm import tqdm

from ...config import settings


def pick_device(torch: Any) -> str:
    """The fastest available device: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """Encode texts into normalized sentence vectors (mean pooling)."""

    max_tokens = 256

    def __init__(self, model_name: str = settings.embedding_model) -> None:
        """Load the tokenizer and the encoder (lazily imports torch)."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch: Any = torch
        self.device = pick_device(torch)
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        model: Any = AutoModel.from_pretrained(model_name)
        self.model: Any = model.to(self.device)
        self.model.eval()

    def encode(self, texts: List[str], batch_size: int = 64,
               progress: bool = False) -> np.ndarray:
        """Return one L2-normalized vector per text, shape (n, dim).

        Each text is truncated to ``max_tokens`` tokens; the vector is the
        mean of the token vectors (padding excluded).
        """
        vectors: List[np.ndarray] = []
        for start in tqdm(range(0, len(texts), batch_size),
                          desc="Embedding chunks", unit="batch",
                          disable=not progress):
            batch = texts[start:start + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=self.max_tokens,
                                    return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(dim=1) \
                / mask.sum(dim=1).clamp(min=1e-9)
            normed = self.torch.nn.functional.normalize(mean, p=2, dim=1)
            vectors.append(normed.cpu().numpy().astype(np.float32))
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(vectors)


class EmbedderProvider:
    """Creates encoders; tests swap ``factory`` for a fast fake."""

    default_model = settings.embedding_model

    def __init__(self, factory: Callable[[str], Any] = Embedder) -> None:
        """Remember how to build an encoder."""
        self.factory = factory

    def create(self, model_name: str = settings.embedding_model) -> Any:
        """A new encoder for ``model_name`` (loads the weights)."""
        return self.factory(model_name)


embedder_provider = EmbedderProvider()
