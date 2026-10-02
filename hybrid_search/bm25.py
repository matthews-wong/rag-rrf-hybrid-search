import math
from collections import Counter

from .tokenize import tokenize

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class BM25Index:
    def __init__(self, docs: dict[str, str], k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self._k1 = k1
        self._b = b
        self._tf = {doc_id: Counter(tokenize(text)) for doc_id, text in docs.items()}
        self._len = {doc_id: sum(tf.values()) for doc_id, tf in self._tf.items()}
        self._avg_len = sum(self._len.values()) / len(docs) if docs else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf.values():
            df.update(tf.keys())
        n = len(docs)
        # The +1 inside the log keeps idf positive for terms in most documents.
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str) -> list[tuple[str, float]]:
        """Return (doc_id, score) for every matching document, best first."""
        terms = tokenize(query)
        scored = []
        for doc_id, tf in self._tf.items():
            score = 0.0
            for t in terms:
                freq = tf.get(t, 0)
                if not freq:
                    continue
                norm = 1 - self._b + self._b * self._len[doc_id] / self._avg_len
                score += self._idf[t] * freq * (self._k1 + 1) / (freq + self._k1 * norm)
            if score > 0:
                scored.append((doc_id, score))
        # Ties break on id so rankings are deterministic.
        return sorted(scored, key=lambda p: (-p[1], p[0]))
