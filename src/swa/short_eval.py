"""Short-context knowledge / reasoning benchmarks from Table 2.

Wraps lm-evaluation-harness to evaluate a SWA-patched HF causal LM on the six
benchmarks used in the paper:

* MMLU (5-shot)
* ARC-C, ARC-Easy
* HellaSwag
* PIQA
* WinoGrande

Usage (programmatic)::

    from swa.short_eval import run_short_eval
    results = run_short_eval(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        window_size=64, num_sinks=4,
        tasks=["mmlu", "arc_easy", "hellaswag", "piqa", "winogrande"],
    )

Or via the CLI::

    python -m swa.run short --model <id> --window 64 --sinks 4 --tasks mmlu,...
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from .patch import patch_swa


# The six benchmarks from the paper's Table 2.
DEFAULT_TASKS = ["mmlu", "arc_challenge", "arc_easy", "hellaswag", "piqa", "winogrande"]


@dataclass
class ShortEvalConfig:
    model_id: str
    window_size: int = 64
    num_sinks: int = 4
    tasks: list[str] | None = None
    batch_size: str = "auto:4"
    num_fewshot: int | None = None
    limit: int | None = None
    output_path: Path | None = None
    dtype: str = "float16"


def _build_lm(model_id: str, window_size: int, num_sinks: int, dtype: str):
    """Load a HF LM, patch it with SWA, and wrap it for lm-eval-harness."""
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model = patch_swa(model, window_size=window_size, num_sinks=num_sinks)
    return HFLM(pretrained=model, tokenizer=tok, batch_size="auto:4")


def run_short_eval(
    *,
    model_id: str,
    window_size: int = 64,
    num_sinks: int = 4,
    tasks: Iterable[str] | None = None,
    batch_size: str = "auto:4",
    num_fewshot: int | None = None,
    limit: int | None = None,
    output_path: Path | None = None,
    dtype: str = "float16",
) -> dict[str, float]:
    """Run short-context benchmarks against a SWA-patched model.

    Returns a dict mapping task name -> accuracy (or primary metric).
    """
    import lm_eval

    task_list = list(tasks) if tasks is not None else list(DEFAULT_TASKS)
    lm = _build_lm(model_id, window_size, num_sinks, dtype)
    # HFLM's batch_size was set in constructor; allow override
    lm.batch_size = batch_size

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=num_fewshot,
        limit=limit,
        bootstrap_iters=10,
        log_samples=False,
    )

    flat: dict[str, float] = {}
    for task, res in results["results"].items():
        # lm-eval reports several metrics; pick the canonical accuracy-ish one.
        for key in ("acc,none", "acc_norm,none", "mc2,none", "acc"):
            if key in res:
                flat[task] = float(res[key])
                break

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(
            {
                "config": {
                    "model_id": model_id,
                    "window_size": window_size,
                    "num_sinks": num_sinks,
                    "tasks": task_list,
                    "num_fewshot": num_fewshot,
                    "limit": limit,
                },
                "results": flat,
            },
            indent=2,
        ))

    return flat


def _cli() -> None:
    p = argparse.ArgumentParser(description="Run short-context SWA evals (Table 2)")
    p.add_argument("--model", required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--sinks", type=int, default=4)
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    p.add_argument("--batch-size", default="auto:4")
    p.add_argument("--num-fewshot", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    results = run_short_eval(
        model_id=args.model,
        window_size=args.window,
        num_sinks=args.sinks,
        tasks=[t.strip() for t in args.tasks.split(",") if t.strip()],
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        output_path=args.out,
        dtype=args.dtype,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    _cli()