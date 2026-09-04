"""BABILong benchmark from Table 4.

The paper evaluates SWA vs LoLCATs on the **average of QA1-QA5** of BABILong
at context lengths 0K–4K. We re-implement a minimal bAbI QA1–QA5 generator
(no external ``babilong`` dependency) and place each story inside a long
filler haystack.

Task summary (bAbI):

* QA1: Single supporting fact (locations).
* QA2: Two supporting facts (travel between locations).
* QA3: Three supporting facts (object movement).
* QA4: Yes/No questions (where an object is now located).
* QA5: Counting / listing (what an actor is holding).

Each task is a callable in :data:`TASK` that returns
``(story, question, answer)``. Following the paper, score = fraction of
greedy completions whose text contains the answer (case- and
whitespace-insensitive). Reported accuracy is the mean across QA1..QA5.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from .load import load
from .patch import Config
from .sample import Sample, contains, filler
from .seed import seeded_rng


NAMES = [
    "Mary", "John", "Sandra", "Daniel", "Emily", "Tom", "Anna", "Peter",
    "Lisa", "Carlos", "Bob", "Alice", "Frank", "Grace", "Henry",
]
PLACES = [
    "kitchen", "garden", "office", "bedroom", "hallway", "bathroom",
    "park", "school", "library", "restaurant", "shop", "market", "lobby",
]
OBJECT = [
    "apple", "banana", "football", "book", "pen", "key", "ball", "milk",
    "toy", "phone", "knife", "ring", "letter", "watch", "glasses",
]


def qa1(rng: random.Random) -> tuple[str, str, str]:
    name = rng.choice(NAMES)
    loc = rng.choice(PLACES)
    return f"{name} is in the {loc}.", f"Where is {name}?", loc


def qa2(rng: random.Random) -> tuple[str, str, str]:
    p1, p2 = rng.sample(PLACES, 2)
    actor = rng.choice(NAMES)
    return (
        f"{actor} went to the {p1}. Then {actor} travelled to the {p2}.",
        f"Where is {actor}?",
        p2,
    )


def qa3(rng: random.Random) -> tuple[str, str, str]:
    p1, p2 = rng.sample(PLACES, 2)
    actor = rng.choice(NAMES)
    obj = rng.choice(OBJECT)
    return (
        f"{actor} picked up the {obj} from the {p1}. {actor} went to the {p2}. {actor} dropped the {obj}.",
        f"Where is the {obj}?",
        p2,
    )


def qa4(rng: random.Random) -> tuple[str, str, str]:
    p1, p2 = rng.sample(PLACES, 2)
    actor = rng.choice(NAMES)
    obj = rng.choice(OBJECT)
    return (
        f"The {obj} is in the {p1}. {actor} picked up the {obj}. {actor} went to the {p2}.",
        f"Is {actor} in the {p1}?",
        "no",
    )


def qa5(rng: random.Random) -> tuple[str, str, str]:
    actor = rng.choice(NAMES)
    items = rng.sample(OBJECT, rng.randint(2, 4))
    return (
        " ".join(f"{actor} picked up the {x}." for x in items),
        f"What is {actor} holding?",
        " ".join(items),
    )


TASK: dict[int, callable] = {1: qa1, 2: qa2, 3: qa3, 4: qa4, 5: qa5}


def samples(
    *,
    task: list[int],
    ctx: list[int],
    frac: list[float] | None = None,
    seed: int = 0,
    n: int = 3,
) -> list[Sample]:
    """Generate BABILong samples across (task, ctx, frac)."""
    if frac is None:
        frac = [0.5]
    bad = set(task) - set(TASK)
    if bad:
        raise ValueError(f"unknown task ids: {bad}; choose from {list(TASK)}")

    rng = seeded_rng(seed)
    out: list[Sample] = []
    for qa in task:
        gen = TASK[qa]
        for c in ctx:
            for f in frac:
                for _ in range(n):
                    story, question, answer = gen(rng)
                    hay = filler(rng, max(1, c - len(story.split())))
                    tokens = hay.split()
                    pos = max(0, min(len(tokens) - 1, int(len(tokens) * f)))
                    tokens.insert(pos, story)
                    prompt = " ".join(tokens) + f"\n\n{question}\nAnswer:"
                    out.append(Sample(prompt=prompt, answer=answer, meta={"task": qa, "ctx": c}))
    return out


def run(
    *,
    model: str,
    cfg: Config,
    ctx: list[int],
    task: list[int] | None = None,
    frac: list[float] | None = None,
    max_new_tokens: int = 32,
    seed: int = 0,
    n: int = 3,
    dtype: torch.dtype | str = torch.float16,
    max_ctx: int | None = None,
    out: Path | None = None,
) -> dict[int, dict[int, float]]:
    """Run BABILong. Returns ``{ctx: {task: acc}}``.

    ``max_ctx`` bounds the token length of each input prompt. When ``None``
    prompts are not truncated at all. When set, prompts longer than ``max_ctx``
    tokens are truncated (which may clip the question or answer — use it
    deliberately, e.g. to fit a fixed context window).
    """
    if task is None:
        task = list(TASK)

    m, tok = load(model, cfg, dtype=dtype)

    correct: dict[tuple[int, int], int] = defaultdict(int)
    total: dict[tuple[int, int], int] = defaultdict(int)

    cell = samples(task=task, ctx=ctx, frac=frac, seed=seed, n=n)
    for s in cell:
        if max_ctx is None:
            ids = tok(s.prompt, return_tensors="pt").input_ids.to(m.device)
        else:
            ids = tok(
                s.prompt, return_tensors="pt", truncation=True, max_length=max_ctx
            ).input_ids.to(m.device)
        with torch.no_grad():
            out_ids = m.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        completion = tok.decode(out_ids[0, ids.shape[1]:], skip_special_tokens=True)
        key_t = (s.meta["ctx"], s.meta["task"])
        total[key_t] += 1
        if contains(s.answer, completion):
            correct[key_t] += 1

    by_ctx: dict[int, dict[int, float]] = {c: {q: 0.0 for q in task} for c in ctx}
    for (c, q), n_total in total.items():
        by_ctx[c][q] = correct[(c, q)] / max(1, n_total)

    for c in ctx:
        avg = sum(by_ctx[c][q] for q in task) / len(task)
        print(f"BABILong ctx={c} avg={avg:.3f} per-task={by_ctx[c]}")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({f"ctx_{c}_task_{q}": acc for c, qm in by_ctx.items() for q, acc in qm.items()}, indent=2))
    return by_ctx


__all__ = ["TASK", "samples", "run"]