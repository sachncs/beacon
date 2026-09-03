"""Single Needle-in-a-Haystack (S-NIAH) benchmark from Table 3.

We evaluate retrieval across three S-NIAH variants:

* **V1**: A simple factual statement.  e.g. ``"The best thing to do in San
  Francisco is to eat a sandwich at the restaurant."`` → ``"sandwich"``
* **V2**: An identifier (a UUID-like value).
* **V3**: A multi-hop variant where the needle encodes a key→value pair and the
  question asks for the value of a *different* key that requires looking up two
  facts.

Each variant is a callable in :data:`VARIANT` that returns
``(needle, question, answer)``. Following the paper, accuracy is the fraction
of greedy-decoded completions whose text contains the ground-truth answer
(case- and whitespace-insensitive).
"""

from __future__ import annotations

import argparse
import json
import random
import string
from collections import defaultdict
from pathlib import Path

import torch

from .load import load
from .patch import Config
from .sample import Sample, contains, filler, insert


# V1: constant needle / question / answer.
NEEDLE = "The best thing to do in San Francisco is to eat a sandwich at the restaurant."
QUESTION = "What is the best thing to do in San Francisco?"
ANSWER = "sandwich"


def key(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))


def v1(rng: random.Random) -> tuple[str, str, str]:
    return NEEDLE, QUESTION, ANSWER


def v2(rng: random.Random) -> tuple[str, str, str]:
    k = key(rng)
    magic = str(rng.randint(10**8, 10**9))
    return f"The special magic number for {k} is: {magic}", f"What is the special magic number for {k}?", magic


def v3(rng: random.Random) -> tuple[str, str, str]:
    a, va = key(rng), "".join(rng.choices(string.ascii_lowercase, k=10))
    b, vb = key(rng), "".join(rng.choices(string.ascii_lowercase, k=10))
    return (
        f"If {a} then {va}. If {b} then {vb}. ",
        f"Question: {vb} is the value of which key?",
        b,
    )


VARIANT: dict[str, callable] = {"1": v1, "2": v2, "3": v3}


def samples(
    *,
    ctx: list[int],
    frac: list[float] | None = None,
    variant: str = "1",
    seed: int = 0,
    n: int = 1,
) -> list[Sample]:
    """Generate NIAH samples for one variant over the (ctx, frac) grid."""
    if frac is None:
        frac = [0.0, 0.25, 0.5, 0.75, 1.0]
    if variant not in VARIANT:
        raise ValueError(f"unknown variant: {variant!r}; choose from {list(VARIANT)}")

    gen = VARIANT[variant]
    rng = random.Random(seed)
    out: list[Sample] = []
    for c in ctx:
        for f in frac:
            for _ in range(n):
                hay = filler(rng, c)
                needle, question, answer = gen(rng)
                body = insert(hay, needle, f)
                prompt = f"{body}\n\n{question}\nAnswer:"
                out.append(Sample(prompt=prompt, answer=answer, meta={"variant": variant, "ctx": c}))
    return out


def run(
    *,
    model: str,
    cfg: Config,
    ctx: list[int],
    variant: list[str] | None = None,
    frac: list[float] | None = None,
    max: int = 32,
    seed: int = 0,
    n: int = 1,
    dtype: torch.dtype | str = torch.float16,
    out: Path | None = None,
) -> dict[tuple[str, int], float]:
    """Run NIAH over the (variant, ctx_len) grid. Returns ``{(variant, ctx): acc}``."""
    if variant is None:
        variant = list(VARIANT)

    m, tok = load(model, cfg, dtype=dtype)

    # Aggregate accuracy per (variant, ctx) cell.
    correct: dict[tuple[str, int], int] = defaultdict(int)
    total: dict[tuple[str, int], int] = defaultdict(int)

    for v in variant:
        cell = samples(ctx=ctx, frac=frac, variant=v, seed=seed, n=n)
        for s in cell:
            ids = tok(s.prompt, return_tensors="pt").input_ids.to(m.device)
            with torch.no_grad():
                out_ids = m.generate(
                    ids,
                    max_new_tokens=max,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            completion = tok.decode(out_ids[0, ids.shape[1]:], skip_special_tokens=True)
            key_t = (v, s.meta["ctx"])
            total[key_t] += 1
            if contains(s.answer, completion):
                correct[key_t] += 1

    results = {k: correct[k] / max(1, total[k]) for k in total}
    for (v, c), acc in results.items():
        print(f"NIAH-{v} ctx={c}: acc={acc:.3f}")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({f"{v}_{c}": acc for (v, c), acc in results.items()}, indent=2))
    return results


def cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run NIAH (Table 3)")
    p.add_argument("--model", required=True)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--sink", type=int, default=4)
    p.add_argument("--ctx", default="512,1024,2048,4096")
    p.add_argument("--variant", default="1,2,3")
    p.add_argument("--frac", default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--max", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    run(
        model=args.model,
        cfg=Config(window=args.window, sink=args.sink),
        ctx=[int(x) for x in args.ctx.split(",")],
        variant=[x.strip() for x in args.variant.split(",") if x],
        frac=[float(x) for x in args.frac.split(",") if x],
        max=args.max,
        seed=args.seed,
        n=args.n,
        dtype=args.dtype,
        out=args.out,
    )
    return 0


__all__ = ["VARIANT", "samples", "run", "cli"]


if __name__ == "__main__":
    raise SystemExit(cli())