"""Short-context knowledge / reasoning benchmarks from Table 2.

Wraps lm-evaluation-harness to evaluate a SWA-patched HF causal LM on the six
benchmarks used in the paper:

* MMLU (5-shot)
* ARC-C, ARC-Easy
* HellaSwag
* PIQA
* WinoGrande

Usage (programmatic)::

    from beacon.short import Short, run
    results = run(Short(model="openbmb/MiniCPM5-1B",
                        cfg=Config(window=64, sink=4)))

Or via the CLI::

    python -m beacon.cli short --model <id> --window 64 --sink 4
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM

from .load import load
from .patch import Config


DEFAULT_TASK: tuple[str, ...] = ("mmlu", "arc_challenge", "arc_easy", "hellaswag", "piqa", "winogrande")


@dataclass(frozen=True)
class Short:
    """Configuration for a short-context evaluation run."""

    model: str
    cfg: Config
    task: tuple[str, ...] = DEFAULT_TASK
    batch: str = "auto:4"
    shot: int = 5
    limit: int | None = None
    dtype: torch.dtype | str = torch.float16
    out: Path | None = None


def run(s: Short) -> dict[str, float]:
    """Run the configured benchmarks. Returns ``{task: primary_metric}``."""
    m, tok = load(s.model, s.cfg, dtype=s.dtype)

    # HFLM takes the patched model directly; no wrapper class needed.
    lm = HFLM(pretrained=m, tokenizer=tok, batch_size=s.batch)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=list(s.task),
        num_fewshot=s.shot,
        limit=s.limit,
        bootstrap_iters=10,
        log_samples=False,
    )

    # lm-eval reports several metrics per task; pick the canonical accuracy-ish one.
    flat: dict[str, float] = {}
    for task, res in results["results"].items():
        for key in ("acc,none", "acc_norm,none", "mc2,none", "acc"):
            if key in res:
                flat[task] = float(res[key])
                break

    if s.out is not None:
        s.out.parent.mkdir(parents=True, exist_ok=True)
        s.out.write_text(json.dumps(
            {
                "config": {
                    "model": s.model,
                    "window": s.cfg.window,
                    "sink": s.cfg.sink,
                    "task": list(s.task),
                    "shot": s.shot,
                    "limit": s.limit,
                },
                "results": flat,
            },
            indent=2,
        ))
    return flat


__all__ = ["Short", "DEFAULT_TASK", "run"]