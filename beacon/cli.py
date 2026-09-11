"""Unified CLI for the SWA-beats-linear reproduction.

Subcommands:

* ``short``  — Table 2 benchmarks (MMLU, ARC, HellaSwag, PIQA, WinoGrande)
* ``find``   — Table 3  (Single Needle-in-a-Haystack)
* ``story``  — Table 4  (BABILong)
* ``speed``  — Figure 2 (throughput / KV-cache memory)

Each subcommand dispatches to a per-module ``run(...)`` function via a small
builder. No ``_run`` glue, no ``flag_to_key`` mapping, no fragment duplication.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from . import find, short, speed, story
from .patch import Config

DEFAULT_MODEL = os.environ.get("BEACON_MODEL", "openbmb/MiniCPM5-1B")

DEFAULT_WINDOW = 64
DEFAULT_SINK = 4


def optional_path(s: str | None) -> Path | None:
    return Path(s) if s else None


def ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def strs(s: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def make_config(args: argparse.Namespace):
    return Config(window=args.window, sink=args.sink)


def run_short(args) -> None:
    s = short.Short(
        model=args.model,
        cfg=make_config(args),
        task=strs(args.task),
        batch=args.batch,
        shot=args.shot,
        limit=args.limit,
        dtype=args.dtype,
        out=optional_path(args.out),
    )
    print(json.dumps(short.run(s), indent=2))


def run_find(args) -> None:
    find.run(
        model=args.model,
        cfg=make_config(args),
        ctx=ints(args.ctx),
        variant=strs(args.variant),
        frac=floats(args.frac),
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        n=args.n,
        dtype=args.dtype,
        out=optional_path(args.out),
    )


def run_story(args) -> None:
    story.run(
        model=args.model,
        cfg=make_config(args),
        ctx=ints(args.ctx),
        task=ints(args.task),
        frac=floats(args.frac),
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        n=args.n,
        dtype=args.dtype,
        max_ctx=args.max_ctx,
        out=optional_path(args.out),
    )


def run_speed(args) -> None:
    speed.run(
        model=args.model,
        cfg=make_config(args),
        ctx=ints(args.ctx),
        method=strs(args.method),
        step=args.step,
        seed=args.seed,
        dtype=args.dtype,
        out=optional_path(args.out),
    )


dispatch: dict[str, Callable[[argparse.Namespace], None]] = {
    "short": run_short,
    "find": run_find,
    "story": run_story,
    "speed": run_speed,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="beacon-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("short", help="short-context benchmarks (Table 2)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--sink", type=int, default=DEFAULT_SINK)
    p.add_argument("--task", default="mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande")
    p.add_argument("--batch", default="auto:4")
    p.add_argument("--shot", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", default=None)

    p = sub.add_parser("find", help="Single Needle-in-a-Haystack (Table 3)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--sink", type=int, default=DEFAULT_SINK)
    p.add_argument("--ctx", default="512,1024,2048,4096")
    p.add_argument("--variant", default="1,2,3")
    p.add_argument("--frac", default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", default=None)

    p = sub.add_parser("story", help="BABILong (Table 4)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--sink", type=int, default=DEFAULT_SINK)
    p.add_argument("--ctx", default="1000,2000,4000,8000")
    p.add_argument("--task", default="1,2,3,4,5")
    p.add_argument("--frac", default="0.5")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-ctx", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", default=None)

    p = sub.add_parser("speed", help="Speed / KV-cache memory benchmark (Figure 2)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--sink", type=int, default=DEFAULT_SINK)
    p.add_argument("--ctx", default="128,256,512,1024,2048,4096,8192,16384,65536")
    p.add_argument("--method", default="fa,swa")
    p.add_argument("--step", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    dispatch[args.cmd](args)
    return 0


__all__ = [
    "main",
    "DEFAULT_MODEL",
    "DEFAULT_WINDOW",
    "DEFAULT_SINK",
    "dispatch",
    "make_config",
    "optional_path",
    "ints",
    "floats",
    "strs",
]


if __name__ == "__main__":
    sys.exit(main())
