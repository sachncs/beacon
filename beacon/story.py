"""BABILong benchmark from Table 4.

The paper evaluates SWA vs LoLCATs on the **average of QA1-QA5** of the BABILong
benchmark at context lengths 0K–4K. We re-implement a minimal bAbI QA1–QA5
generator (no external ``babilong`` dependency) and place the bAbI story inside
a long filler "haystack" identical in structure to the original BABILong setup.

Task summary (bAbI):

* QA1: Single supporting fact (locations).
* QA2: Two supporting facts (travel between locations).
* QA3: Three supporting facts (object movement).
* QA4: Yes/No questions (where an object is now located).
* QA5: Counting / listing (what an actor is holding).

Following the paper, we score as: 1 if the model's greedy answer contains the
ground truth, 0 otherwise. We report the mean accuracy across QA1..QA5.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from .patch import Config, patch


# Vocabularies
NAMES = [
    "Mary", "John", "Sandra", "Daniel", "Emily", "Tom", "Anna", "Peter",
    "Lisa", "Carlos", "Bob", "Alice", "Frank", "Grace", "Henry",
]
PLACES = [
    "kitchen", "garden", "office", "bedroom", "hallway", "bathroom",
    "park", "school", "library", "restaurant", "shop", "market", "lobby",
]
OBJECTS = [
    "apple", "banana", "football", "book", "pen", "key", "ball", "milk",
    "toy", "phone", "knife", "ring", "letter", "watch", "glasses",
]


# ------------------------------------------------------------------
# bAbI-style story generators
# ------------------------------------------------------------------


def _make_qa1(rng: random.Random) -> tuple[str, str, str]:
    name = rng.choice(NAMES)
    loc = rng.choice(PLACES)
    story = f"{name} is in the {loc}."
    question = f"Where is {name}?"
    return story, question, loc


def _make_qa2(rng: random.Random) -> tuple[str, str, str]:
    p1, p2 = rng.sample(PLACES, 2)
    actor = rng.choice(NAMES)
    story = (
        f"{actor} went to the {p1}. "
        f"Then {actor} travelled to the {p2}."
    )
    question = f"Where is {actor}?"
    return story, question, p2


def _make_qa3(rng: random.Random) -> tuple[str, str, str]:
    p1, p2 = rng.sample(PLACES, 2)
    actor = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    story = (
        f"{actor} picked up the {obj} from the {p1}. "
        f"{actor} went to the {p2}. "
        f"{actor} dropped the {obj}."
    )
    question = f"Where is the {obj}?"
    return story, question, p2


def _make_qa4(rng: random.Random) -> tuple[str, str, str]:
    p1, p2 = rng.sample(PLACES, 2)
    actor = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    story = (
        f"The {obj} is in the {p1}. "
        f"{actor} picked up the {obj}. "
        f"{actor} went to the {p2}."
    )
    question = f"Is {actor} in the {p1}?"
    return story, question, "no"


def _make_qa5(rng: random.Random) -> tuple[str, str, str]:
    actor = rng.choice(NAMES)
    items = rng.sample(OBJECTS, rng.randint(2, 4))
    story = " ".join(f"{actor} picked up the {x}." for x in items)
    question = f"What is {actor} holding?"
    return story, question, " ".join(items)


QA_GENERATORS: dict[int, Callable[[random.Random], tuple[str, str, str]]] = {
    1: _make_qa1,
    2: _make_qa2,
    3: _make_qa3,
    4: _make_qa4,
    5: _make_qa5,
}


# ------------------------------------------------------------------
# Haystack filler
# ------------------------------------------------------------------

_FILLER_VOCAB = (
    "the of and to in a is that for on with as it was by an be this are not from "
    "at or have but his they she which we one all there their what when your can "
    "said about would been if more her than them no time only do some could so my "
    "these other into make them then like over also our who has been"
).split()


def _filler_text(rng: random.Random, n_words: int) -> str:
    out: list[str] = []
    total = 0
    while total < n_words:
        sent_len = rng.randint(6, 14)
        sent = " ".join(rng.choice(_FILLER_VOCAB) for _ in range(sent_len)) + "."
        out.append(sent.capitalize())
        total += sent_len + 1
    return " ".join(out)


# ------------------------------------------------------------------
# Sample assembly
# ------------------------------------------------------------------


@dataclass
class BABILongSample:
    qa_id: int
    context_words: int
    insert_frac: float
    prompt: str
    answer: str


def make_babilong_samples(
    *,
    qa_ids: list[int],
    context_word_lens: list[int],
    insert_fracs: list[float] | None = None,
    seed: int = 0,
    n_per_cell: int = 3,
) -> list[BABILongSample]:
    if insert_fracs is None:
        insert_fracs = [0.5]
    rng = random.Random(seed)
    samples: list[BABILongSample] = []
    for qa in qa_ids:
        gen = QA_GENERATORS[qa]
        for ctx in context_word_lens:
            for frac in insert_fracs:
                for _ in range(n_per_cell):
                    story, question, answer = gen(rng)
                    haystack = _filler_text(rng, max(1, ctx - len(story.split())))
                    tokens = haystack.split()
                    pos = max(0, min(len(tokens) - 1, int(len(tokens) * frac)))
                    tokens.insert(pos, story)
                    prompt = " ".join(tokens) + f"\n\n{question}\nAnswer:"
                    samples.append(BABILongSample(qa, ctx, frac, prompt, answer))
    return samples


# ------------------------------------------------------------------
# Run + score
# ------------------------------------------------------------------


def _score(answer: str, completion: str) -> bool:
    a = re.sub(r"\W+", " ", answer.lower()).strip()
    c = re.sub(r"\W+", " ", completion.lower()).strip()
    return a in c


def run_babilong(
    *,
    model_id: str,
    window_size: int,
    num_sinks: int,
    context_word_lens: list[int],
    qa_ids: list[int] | None = None,
    insert_fracs: list[float] | None = None,
    max_new_tokens: int = 32,
    dtype: str = "float16",
) -> dict[int, dict[int, float]]:
    """Run BABILong. Returns ``{ctx_len: {qa_id: acc}}``."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if qa_ids is None:
        qa_ids = list(QA_GENERATORS.keys())

    cfg = Config(window=window_size, sink=num_sinks)
    torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model = patch(model, cfg)
    model.eval()

    samples = make_babilong_samples(
        qa_ids=qa_ids,
        context_word_lens=context_word_lens,
        insert_fracs=insert_fracs,
    )

    by_ctx: dict[int, dict[int, float]] = {c: {q: 0.0 for q in qa_ids} for c in context_word_lens}
    counts: dict[tuple[int, int], int] = {(c, q): 0 for c in context_word_lens for q in qa_ids}

    for s in samples:
        ids = tok(s.prompt, return_tensors="pt", truncation=True, max_length=8192).input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        completion = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        counts[(s.context_words, s.qa_id)] += 1
        if _score(s.answer, completion):
            by_ctx[s.context_words][s.qa_id] += 1

    for (c, q), n in counts.items():
        by_ctx[c][q] /= max(1, n)

    for c in context_word_lens:
        avg = sum(by_ctx[c][q] for q in qa_ids) / len(qa_ids)
        print(f"BABILong ctx={c} avg={avg:.3f} per-task={by_ctx[c]}")
    return by_ctx


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _cli() -> None:
    p = argparse.ArgumentParser(description="Run BABILong (Table 4)")
    p.add_argument("--model", required=True)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--sinks", type=int, default=4)
    p.add_argument("--ctxs", default="1000,2000,4000,8000")
    p.add_argument("--qa-ids", default="1,2,3,4,5")
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    ctxs = [int(x) for x in args.ctxs.split(",")]
    qa_ids = [int(x) for x in args.qa_ids.split(",")]

    res = run_babilong(
        model_id=args.model,
        window_size=args.window,
        num_sinks=args.sinks,
        context_word_lens=ctxs,
        qa_ids=qa_ids,
        max_new_tokens=args.max_new,
        dtype=args.dtype,
    )
    flat = {f"ctx_{c}_qa_{q}": acc for c, qm in res.items() for q, acc in qm.items()}
    print(json.dumps(flat, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(flat, indent=2))


if __name__ == "__main__":
    _cli()