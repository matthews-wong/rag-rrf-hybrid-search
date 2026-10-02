import re

_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be by do for from how i in is it of on or the to what when with".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]
