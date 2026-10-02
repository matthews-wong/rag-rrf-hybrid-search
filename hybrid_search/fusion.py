from collections import defaultdict

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]], k: int = DEFAULT_RRF_K
) -> list[tuple[str, float]]:
    """Merge ranked lists using only rank positions, never the raw scores."""
    if k <= 0:
        raise ValueError(f"rrf k must be positive, got {k}")
    fused: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for position, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + position)
    return sorted(fused.items(), key=lambda p: (-p[1], p[0]))
