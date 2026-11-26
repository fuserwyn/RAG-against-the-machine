"""The BM25 matrix files of the native index.

    bm25_weights.npz   W: BM25 weight of every (chunk, term)
    bm25_counts.npz    C: raw term counts (for incremental updates)
    bm25_vocab.json    term -> column id, plus k1 and b
"""

import json
import os

from scipy import sparse

from ..models.bm25_index import BM25Index


class BM25Repository:
    """Saves and loads a :class:`BM25Index`."""

    weights_file = "bm25_weights.npz"
    counts_file = "bm25_counts.npz"
    vocab_file = "bm25_vocab.json"

    def save(self, index: BM25Index, index_dir: str) -> None:
        """Persist the index into ``index_dir``."""
        os.makedirs(index_dir, exist_ok=True)
        sparse.save_npz(os.path.join(index_dir, self.weights_file),
                        index.weights)
        sparse.save_npz(os.path.join(index_dir, self.counts_file),
                        index.counts)
        with open(os.path.join(index_dir, self.vocab_file), "w",
                  encoding="utf-8") as handle:
            json.dump({"k1": index.k1, "b": index.b, "vocab": index.vocab},
                      handle)

    def load(self, index_dir: str, with_counts: bool = False) -> BM25Index:
        """Load an index previously written by :meth:`save`.

        Args:
            index_dir: Index directory.
            with_counts: Also load the raw term counts (only needed to
                update the index incrementally).
        """
        with open(os.path.join(index_dir, self.vocab_file),
                  encoding="utf-8") as handle:
            meta = json.load(handle)
        index = BM25Index(k1=float(meta["k1"]), b=float(meta["b"]))
        index.vocab = {str(k): int(v) for k, v in meta["vocab"].items()}
        index.weights = sparse.load_npz(
            os.path.join(index_dir, self.weights_file)).tocsc()
        if with_counts:
            index.counts = sparse.load_npz(
                os.path.join(index_dir, self.counts_file)).tocsr()
        return index


bm25_repository = BM25Repository()
