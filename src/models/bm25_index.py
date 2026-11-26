"""The BM25 (Okapi) index: a sparse (chunks x terms) weight matrix.

All BM25 weights are computed once, at indexing time, so scoring a query
is only a sum of a few columns:

    score(d, q) = sum_t idf(t) * tf(t, d) * (k1 + 1)
                  / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

The raw term counts are kept too: incremental indexing replaces the rows
of the changed files and recomputes the weights without re-tokenizing the
rest of the corpus. Reading and writing the files is the job of
``repositories.bm25_repository``.
"""

from collections import Counter
from typing import Dict, List, Sequence

import numpy as np
from scipy import sparse


class BM25Index:
    """Vocabulary, term counts and BM25 weights of a tokenized corpus."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        """Create an empty index.

        Args:
            k1: Term-frequency saturation parameter.
            b: Length-normalization strength (0 = none, 1 = full).
        """
        self.k1 = k1
        self.b = b
        self.vocab: Dict[str, int] = {}               # term -> column id
        self.counts: sparse.csr_matrix = sparse.csr_matrix((0, 0))
        self.weights: sparse.csc_matrix = sparse.csc_matrix((0, 0))

    @property
    def n_documents(self) -> int:
        """Number of rows (chunks) of the weight matrix."""
        return int(self.weights.shape[0])

    def build(self, documents: Sequence[List[str]]) -> None:
        """Compute the BM25 weight matrix for tokenized documents."""
        self.vocab = {}
        self.set_counts(self.count_matrix(documents))

    def count_matrix(self, documents: Sequence[List[str]]
                     ) -> sparse.csr_matrix:
        """Return the (documents x terms) count matrix, growing the vocab.

        The matrix has one column per term currently in the vocabulary, so
        it can be stacked under the counts of an older index.
        """
        rows: List[int] = []
        cols: List[int] = []
        tfs: List[int] = []
        for doc_id, tokens in enumerate(documents):
            for term, count in Counter(tokens).items():
                rows.append(doc_id)
                cols.append(self.vocab.setdefault(term, len(self.vocab)))
                tfs.append(count)
        return sparse.csr_matrix(
            (np.asarray(tfs, dtype=np.int32),
             (np.asarray(rows, dtype=np.int64),
              np.asarray(cols, dtype=np.int64))),
            shape=(len(documents), len(self.vocab)))

    def set_counts(self, counts: sparse.csr_matrix) -> None:
        """Replace the term counts and recompute every BM25 weight."""
        self.counts = sparse.csr_matrix(counts)
        coo = self.counts.tocoo()
        n_docs, n_terms = self.counts.shape
        tf = coo.data.astype(np.float64)
        lengths = np.asarray(self.counts.sum(axis=1),
                             dtype=np.float64).ravel()
        df = np.bincount(coo.col, minlength=n_terms)
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        avgdl = float(lengths.mean()) if n_docs else 1.0
        norm = self.k1 * (1.0 - self.b + self.b * lengths / max(avgdl, 1.0))
        values = idf[coo.col] * tf * (self.k1 + 1.0) / (tf + norm[coo.row])
        self.weights = sparse.csc_matrix(
            (values.astype(np.float32), (coo.row, coo.col)),
            shape=(n_docs, n_terms))

    def score(self, query_tokens: List[str]) -> np.ndarray:
        """Return the BM25 score of every document for a query."""
        ids = [self.vocab[t] for t in set(query_tokens) if t in self.vocab]
        if not ids:
            return np.zeros(self.n_documents, dtype=np.float32)
        result = self.weights[:, ids].sum(axis=1)
        return np.asarray(result, dtype=np.float32).ravel()
