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
from typing import Any


DEFAULT_MODEL = os.environ.get("BEACON_MODEL", "openbmb/MiniCPM5-1B")


def _path(s: str | None) -> Path | None:
    return Path(s) if s else None


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def _strs(s: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def _cfg(args: argparse.Namespace):
    from .patch import Config

    return Config(window=args.window, sink=args.sink)


def _build_short(args) -> None:
    from .short import Short, run

    s = Short(
        model=args.model,
        cfg=_cfg(args),
        task=_strs(args.task),
        batch=args.batch,
        shot=args.shot,
        limit=args.limit,
        dtype=args.dtype,
        out=_path(args.out),
    )
    print(json.dumps(run(s), indent=2))


def _build_find(args) -> None:
    from .find import run

    run(
        model=args.model,
        cfg=_cfg(args),
        ctx=_ints(args.ctx),
        variant=_strs(args.variant),
        frac=_floats(args.frac),
        max=args.max,
        seed=args.seed,
        n=args.n,
        dtype=args.dtype,
        out=_path(args.out),
    )


def _build_story(args) -> None:
    from .story import run

    run(
        model=args.model,
        cfg=_cfg(args),
        ctx=_ints(args.ctx),
        task=_ints(args.task),
        frac=_floats(args.frac),
        max=args.max,
        seed=args.seed,
        n=args.n,
        dtype=args.dtype,
        out=_path(args.out),
    )


def _build_speed(args) -> None:
    from .speed import run

    run(
        model=args.model,
        cfg=_cfg(args),
        ctx=_ints(args.ctx),
        method=_strs(args.method),
        step=args.step,
        seed=args.seed,
        dtype=args.dtype,
        out=_path(args.out),
    )


_DISPATCH: dict[str, Callable[[argparse.Namespace], None]] = {
    "short": _build_short,
    "find": _build_find,
    "story": _build_story,
    "speed": _build_speed,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="beacon-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("short", help="short-context benchmarks (Table 2)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--sink", type=int, default=4)
    p.add_argument("--task", default="mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande")
    p.add_argument("--batch", default="auto:4")
    p.add_argument("--shot", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", default=None)

    p = sub.add_parser("find", help="Single Needle-in-a-Haystack (Table 3)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--sink", type=int, default=4)
    p.add_argument("--ctx", default="512,1024,2048,4096")
    p.add_argument("--variant", default="1,2,3")
    p.add_argument("--frac", default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--max", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", default=None)

    p = sub.add_parser("story", help="BABILong (Table 4)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--sink", type=int, default=4)
    p.add_argument("--ctx", default="1000,2000,4000,8000")
    p.add_argument("--task", default="1,2,3,4,5")
    p.add_argument("--frac", default="0.5")
    p.add_argument("--max", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", default=None)

    p = sub.add_parser("speed", help="Speed / KV-cache memory benchmark (Figure 2)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--sink", type=int, default=4)
    p.add_argument("--ctx", default="128,256,512,1024,2048,4096,8192,16384,65536")
    p.add_argument("--method", default="fa,swa")
    p.add_argument("--step", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    _DISPATCH[args.cmd](args)
    return 0


__all__ = ["main", "DEFAULT_MODEL"]


if __name__ == "__main__":
    sys.exit(main())