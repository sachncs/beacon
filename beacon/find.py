"""Single Needle-in-a-Haystack (S-NIAH) benchmark from Table 3.

We evaluate retrieval across three S-NIAH variants:

* **S-NIAH-1**: A simple factual statement. ``"The best thing to do in San Francisco is
  to eat a sandwich at the restaurant."``  →  ``Question: What is the best thing to do
  in San Francisco?``
* **S-NIAH-2**: An identifier (a UUID-like value).  → ``Question: What is the special
  magic number for {key}?``
* **S-NIAH-3**: A multi-hop variant where the needle encodes a key→value pair and the
  question asks for the value of a *different* key that requires looking up two facts.

Following the paper, we score accuracy by checking whether the model's greedy-decoded
answer string contains the expected ground-truth substring (case-insensitive).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from .patch import Config, patch


# Three S-NIAH variants used in the paper.
NEEDLE_S1 = "The best thing to do in San Francisco is to eat a sandwich at the restaurant."
QUESTION_S1 = "What is the best thing to do in San Francisco?"
ANSWER_S1 = "sandwich"


def _random_key(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))


def _make_niah2(rng: random.Random) -> tuple[str, str, str]:
    key = _random_key(rng)
    magic = str(rng.randint(10**8, 10**9))
    needle = f"The special magic number for {key} is: {magic}"
    question = f"What is the special magic number for {key}?"
    return needle, question, magic


def _make_niah3(rng: random.Random) -> tuple[str, str, str]:
    key_a = _random_key(rng)
    val_a = "".join(rng.choices(string.ascii_lowercase, k=10))
    key_b = _random_key(rng)
    val_b = "".join(rng.choices(string.ascii_lowercase, k=10))
    needle = (
        f"If {key_a} then {val_a}. "
        f"If {key_b} then {val_b}. "
    )
    question = f"Question: {val_b} is the value of which key?"
    # The answer is key_b itself, requiring multi-hop.
    return needle, question, key_b


def _filler_sentences(rng: random.Random, n_words: int) -> str:
    """Build a haystack of ~n_words pseudo-English filler text."""
    vocab = (
        "the of and to in a is that for on with as it was by an be this are not from "
        "at or have but his they she which we one all there their what when your can "
        "said about would been if more her than them no time only do some could so my "
        "these other into make them then like over also our who has been"
    ).split()
    out = []
    total = 0
    while total < n_words:
        sent_len = rng.randint(6, 14)
        sent = " ".join(rng.choice(vocab) for _ in range(sent_len)) + "."
        out.append(sent.capitalize())
        total += sent_len + 1
    return " ".join(out)


def _build_prompt(haystack: str, needle: str, question: str, insert_frac: float) -> str:
    """Insert ``needle`` at fraction ``insert_frac`` of the haystack."""
    tokens = haystack.split()
    n = len(tokens)
    pos = max(1, min(n - 1, int(n * insert_frac)))
    tokens.insert(pos, needle)
    body = " ".join(tokens)
    return f"{body}\n\n{question}\nAnswer:"


@dataclass
class NIAHSample:
    context_len: int
    insert_frac: float
    prompt: str
    answer: str


def make_niah_samples(
    *,
    context_lens: list[int],
    insert_fracs: list[float] | None = None,
    variant: str = "1",
    seed: int = 0,
    n_per_cell: int = 1,
) -> list[NIAHSample]:
    """Generate S-NIAH samples across (context_len, insert_frac) grid cells."""
    if insert_fracs is None:
        insert_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    rng = random.Random(seed)
    samples: list[NIAHSample] = []

    for ctx in context_lens:
        for frac in insert_fracs:
            for _ in range(n_per_cell):
                haystack = _filler_sentences(rng, ctx)
                if variant == "1":
                    prompt = _build_prompt(haystack, NEEDLE_S1, QUESTION_S1, frac)
                    answer = ANSWER_S1
                elif variant == "2":
                    needle, question, magic = _make_niah2(rng)
                    prompt = _build_prompt(haystack, needle, question, frac)
                    answer = magic
                elif variant == "3":
                    needle, question, key = _make_niah3(rng)
                    prompt = _build_prompt(haystack, needle, question, frac)
                    answer = key
                else:
                    raise ValueError(f"Unknown variant: {variant}")
                samples.append(NIAHSample(ctx, frac, prompt, answer))
    return samples


def _score_contains(answer: str, completion: str) -> bool:
    return answer.lower() in completion.lower()


def run_niah(
    *,
    model_id: str,
    window_size: int,
    num_sinks: int,
    context_lens: list[int],
    variants: list[str] | None = None,
    insert_fracs: list[float] | None = None,
    max_new_tokens: int = 32,
    dtype: str = "float16",
) -> dict[tuple[str, int], float]:
    """Run S-NIAH over the (variant, context_len) grid.

    Returns accuracy per cell as ``{(variant, ctx_len): acc}``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if variants is None:
        variants = ["1", "2", "3"]

    cfg = Config(window=window_size, sink=num_sinks)
    torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model = patch(model, cfg)
    model.eval()

    results: dict[tuple[str, int], float] = {}
    for variant in variants:
        samples = make_niah_samples(
            context_lens=context_lens,
            insert_fracs=insert_fracs,
            variant=variant,
        )
        for ctx in context_lens:
            cell = [s for s in samples if s.context_len == ctx]
            correct = 0
            for s in cell:
                ids = tok(s.prompt, return_tensors="pt").input_ids.to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                    )
                completion = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
                if _score_contains(s.answer, completion):
                    correct += 1
            acc = correct / max(1, len(cell))
            results[(variant, ctx)] = acc
            print(f"S-NIAH-{variant} ctx={ctx}: acc={acc:.3f}")
    return results


def _cli() -> None:
    p = argparse.ArgumentParser(description="Run S-NIAH (Table 3)")
    p.add_argument("--model", required=True)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--sinks", type=int, default=4)
    p.add_argument("--ctxs", default="512,1024,2048,4096")
    p.add_argument("--variants", default="1,2,3")
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    ctxs = [int(x) for x in args.ctxs.split(",") if x]
    variants = [x.strip() for x in args.variants.split(",") if x]

    results = run_niah(
        model_id=args.model,
        window_size=args.window,
        num_sinks=args.sinks,
        context_lens=ctxs,
        variants=variants,
        max_new_tokens=args.max_new,
        dtype=args.dtype,
    )

    out = {f"{v}_{c}": acc for (v, c), acc in results.items()}
    print(json.dumps(out, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    _cli()