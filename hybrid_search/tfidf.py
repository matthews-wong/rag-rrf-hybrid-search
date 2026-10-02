import math
from collections import Counter

from .tokenize import tokenize


class TfidfIndex:
    def __init__(self, docs: dict[str, str]) -> None:
        counts = {doc_id: Counter(tokenize(text)) for doc_id, text in docs.items()}
        df: Counter[str] = Counter()
        for tf in counts.values():
            df.update(tf.keys())
        n = len(docs)
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self._vecs = {doc_id: self._weigh(tf) for doc_id, tf in counts.items()}

    def _weigh(self, tf: Counter[str]) -> dict[str, float]:
        vec = {t: (1 + math.log(c)) * self._idf[t] for t, c in tf.items() if t in self._idf}
        norm = math.sqrt(sum(w * w for w in vec.values()))
        return {t: w / norm for t, w in vec.items()} if norm else {}

    def search(self, query: str) -> list[tuple[str, float]]:
        """Return (doc_id, cosine) for every document sharing a term, best first."""
        q = self._weigh(Counter(tokenize(query)))
        scored = []
        for doc_id, vec in self._vecs.items():
            score = sum(w * vec.get(t, 0.0) for t, w in q.items())
            if score > 0:
                scored.append((doc_id, score))
        return sorted(scored, key=lambda p: (-p[1], p[0]))
