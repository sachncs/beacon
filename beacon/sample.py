"""Shared sample primitives.

Every long-context benchmark is built from the same three ingredients:

* a filler-text haystack
* a needle (story / fact / question) embedded at a fraction of that haystack
* a substring-containment scorer that judges the model's greedy completion

Keeping these in one place eliminates the duplication that previously lived in
``find.py`` and ``story.py``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

_VOCAB = (
    "the of and to in a is that for on with as it was by an be this are not from "
    "at or have but his they she which we one all there their what when your can "
    "said about would been if more her than them no time only do some could so my "
    "these other into make them then like over also our who has been"
).split()


@dataclass(frozen=True)
class Sample:
    """A test prompt with its ground-truth answer and grid-cell metadata.

    ``meta`` carries the keys the runner groups by — e.g. ``{"variant": "1", "ctx": 1024}``
    for NIAH or ``{"task": 3, "ctx": 2000}`` for BABILong.
    """

    prompt: str
    answer: str
    meta: dict


def filler(rng: random.Random, n_words: int) -> str:
    """Pseudo-English haystack of roughly ``n_words`` words.

    The original BABILong / NIAH work uses Project Gutenberg essays as the
    haystack so that the needle is plausibly embedded in real natural
    language. The default here is a synthetic word salad drawn from a small
    English stopword vocabulary, which is enough to exercise the attention
    window and the containment scorer without a multi-MB download. Pass a
    ``rng`` seeded by the caller to keep samples reproducible.
    """
    out: list[str] = []
    total = 0
    while total < n_words:
        sent_len = rng.randint(6, 14)
        sent = " ".join(rng.choice(_VOCAB) for _ in range(sent_len)) + "."
        out.append(sent.capitalize())
        total += sent_len + 1
    return " ".join(out)

def insert(haystack: str, needle: str, frac: float) -> str:
    """Embed ``needle`` at fraction ``frac`` of the word-tokenised haystack."""
    tokens = haystack.split()
    if not tokens:
        return needle
    pos = max(0, min(len(tokens) - 1, int(len(tokens) * frac)))
    tokens.insert(pos, needle)
    return " ".join(tokens)


def _normalize(s: str) -> str:
    """Lowercase and replace every non-letter / non-digit with a single space.

    Underscores are treated as separators so ``hello_world`` matches
    ``hello world`` (Python's ``\\w`` includes ``_``, which would otherwise
    leak into both sides and break the comparison).
    """
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def contains(answer: str, completion: str) -> bool:
    """Case- and whitespace-insensitive containment of ``answer`` in ``completion``."""
    a = _normalize(answer)
    c = _normalize(completion)
    return bool(a) and a in c


__all__ = ["Sample", "filler", "insert", "contains"]
