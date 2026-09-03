"""Unified CLI for the SWA-beats-linear reproduction.

Subcommands:

* ``short``  — Table 2 benchmarks (MMLU, ARC, HellaSwag, PIQA, WinoGrande)
* ``find``   — Table 3  (Single Needle-in-a-Haystack)
* ``story``  — Table 4  (BABILong)
* ``speed``  — Figure 2 (throughput / KV-cache memory)

Each subcommand dispatches to a per-module ``run(...)`` function with a
single config dataclass. No ``_run`` glue, no ``flag_to_key`` mapping.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_MODEL = os.environ.get("BEACON_MODEL", "openbmb/MiniCPM5-1B")


def _build_short(args) -> None:
    from .short import Short, run

    s = Short(
        model=args.model,
        cfg=_cfg(args),
        task=tuple(t.strip() for t in args.task.split(",") if t.strip()),
        batch=args.batch,
        shot=args.shot,
        limit=args.limit,
        dtype=args.dtype,
        out=Path(args.out) if args.out else None,
    )
    import json
    print(json.dumps(run(s), indent=2))


def _build_find(args) -> None:
    from .find import run

    run(
        model=args.model,
        cfg=_cfg(args),
        ctx=[int(x) for x in args.ctx.split(",")],
        variant=[x.strip() for x in args.variant.split(",") if x],
        frac=[float(x) for x in args.frac.split(",") if x],
        max=args.max,
        seed=args.seed,
        n=args.n,
        dtype=args.dtype,
        out=Path(args.out) if args.out else None,
    )


def _build_story(args) -> None:
    from .story import run

    run(
        model=args.model,
        cfg=_cfg(args),
        ctx=[int(x) for x in args.ctx.split(",")],
        task=[int(x) for x in args.task.split(",")],
        frac=[float(x) for x in args.frac.split(",") if x],
        max=args.max,
        seed=args.seed,
        n=args.n,
        dtype=args.dtype,
        out=Path(args.out) if args.out else None,
    )


def _build_speed(args) -> None:
    from .speed import run

    run(
        model=args.model,
        cfg=_cfg(args),
        ctx=[int(x) for x in args.ctx.split(",")],
        method=tuple(x.strip() for x in args.method.split(",") if x),
        step=args.step,
        seed=args.seed,
        dtype=args.dtype,
        out=Path(args.out) if args.out else None,
    )


def _cfg(args):
    from .patch import Config
    return Config(window=args.window, sink=args.sink)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="beacon-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # short
    p = sub.add_parser("short", help="short-context benchmarks (Table 2)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--sink", type=int, default=4)
    p.add_argument("--task", default="mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande")
    p.add_argument("--batch", default="auto:4")
    p.add_argument("--shot", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", type=str, default=None)

    # find (NIAH)
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
    p.add_argument("--out", type=str, default=None)

    # story (BABILong)
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
    p.add_argument("--out", type=str, default=None)

    # speed
    p = sub.add_parser("speed", help="Speed / KV-cache memory benchmark (Figure 2)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--sink", type=int, default=4)
    p.add_argument("--ctx", default="128,256,512,1024,2048,4096,8192,16384,65536")
    p.add_argument("--method", default="fa,swa")
    p.add_argument("--step", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", type=str, default=None)

    args = parser.parse_args(argv)
    dispatch = {
        "short": _build_short,
        "find": _build_find,
        "story": _build_story,
        "speed": _build_speed,
    }
    dispatch[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())