"""Text tokenizer shared by indexing and querying.

Questions paraphrase ideas while code uses identifiers such as
``FusedMoEActivationFormat`` or ``max_num_seqs``. Every identifier is
therefore kept whole *and* split into its sub-words, so a query can match
either the verbatim identifier or the words it is made of.
"""

import re
from typing import List


class Tokenizer:
    """Text -> lower-case, stemmed search terms without stop words."""

    WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
    CAMEL_RE = re.compile(
        r"[A-Z]+(?=[A-Z][a-z0-9])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
    STOPWORDS = frozenset(
        """
        a an and are as at be been being by can could did do does doing for
        from had has have how i if in into is it its itself me my of on or
        our so than that the their them then there these they this those
        through to too very was we were what when where which while who whom
        why will with would you your vllm self
        """.split()
    )

    def stem(self, word: str) -> str:
        """Strip plural and common endings from a lower-case word."""
        if len(word) <= 3:
            return word
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("s") and not word.endswith(("ss", "us", "is")):
            return word[:-1]
        return word

    def split_identifier(self, word: str) -> List[str]:
        """Split a snake_case / camelCase identifier into its parts.

        Returns the lower-case sub-words (a single word gives one part).
        """
        parts: List[str] = []
        for piece in word.split("_"):
            if piece:
                parts.extend(p.lower() for p in self.CAMEL_RE.findall(piece))
        return parts

    def words(self, text: str) -> List[str]:
        """The raw words of a text, before any normalization."""
        return self.WORD_RE.findall(text)

    def tokenize(self, text: str) -> List[str]:
        """Turn any text (question, code, prose) into search terms."""
        tokens: List[str] = []
        for word in self.words(text):
            lower = word.lower()
            if self._keep(lower):
                tokens.append(self.stem(lower))
            parts = self.split_identifier(word)
            if len(parts) > 1:
                tokens.extend(self.stem(p) for p in parts if self._keep(p))
        return tokens

    def _keep(self, word: str) -> bool:
        """Drop stop words and one-letter words."""
        return word not in self.STOPWORDS and len(word) > 1


tokenizer = Tokenizer()
