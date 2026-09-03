"""Unified CLI for the SWA-beats-linear reproduction.

Subcommands:

* ``short``  — Table 2 benchmarks (MMLU, ARC, HellaSwag, PIQA, WinoGrande)
* ``find``   — Table 3  (Single Needle-in-a-Haystack)
* ``story``  — Table 4  (BABILong)
* ``speed``  — Figure 2 (throughput / KV-cache memory)
"""

from __future__ import annotations

import argparse
import os
import sys


DEFAULT_MODEL = os.environ.get("BEACON_MODEL", "openbmb/MiniCPM5-1B")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="beacon-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # short
    p = sub.add_parser("short", help="short-context benchmarks (Table 2)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--sinks", type=int, default=4)
    p.add_argument("--tasks", default="mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande")
    p.add_argument("--batch-size", default="auto:4")
    p.add_argument("--num-fewshot", type=int, default=None)
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

    # story (babilong)
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
    p.add_argument("--sinks", type=int, default=4)
    p.add_argument("--ctxs", default="128,256,512,1024,2048,4096,8192,16384,65536")
    p.add_argument("--decode-steps", type=int, default=64)
    p.add_argument("--methods", default="fa,swa")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", type=str, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "short":
        from .short import run_short_eval

        return _run(run_short_eval, args, [
            "model_id", "window_size", "num_sinks", "tasks",
            "batch_size", "num_fewshot", "limit", "dtype",
        ], out=getattr(args, "out", None))

    if args.cmd == "find":
        from .find import run as run_find
        from .patch import Config

        return _run(run_find, args, [
            "model", "cfg",
            "ctx", "variant", "frac", "max", "seed", "n", "dtype",
        ], out=getattr(args, "out", None))

    if args.cmd == "story":
        from .story import run as run_story
        from .patch import Config

        return _run(run_story, args, [
            "model", "cfg",
            "ctx", "task", "frac", "max", "seed", "n", "dtype",
        ], out=getattr(args, "out", None))

    if args.cmd == "speed":
        from .speed import run_speed_mem

        from pathlib import Path
        kwargs = dict(
            model_id=args.model,
            window_size=args.window,
            num_sinks=args.sinks,
            context_lens=[int(x) for x in args.ctxs.split(",")],
            decode_steps=args.decode_steps,
            methods=[x.strip() for x in args.methods.split(",") if x],
            dtype=args.dtype,
        )
        if args.out:
            kwargs["out"] = Path(args.out)
        run_speed_mem(**kwargs)
        return 0

    parser.error(f"unknown subcommand: {args.cmd}")
    return 2


def _run(fn, args, keys, out=None):
    from pathlib import Path
    from .patch import Config

    # The find and story subcommands take a Config object; build it from flags.
    if args.cmd in ("find", "story"):
        ctx = [int(x) for x in args.ctx.split(",")]
        frac = [float(x) for x in args.frac.split(",") if x]
        if args.cmd == "find":
            kwargs = {
                "model": args.model,
                "cfg": Config(window=args.window, sink=args.sink),
                "ctx": ctx,
                "variant": [x.strip() for x in args.variant.split(",") if x],
                "frac": frac,
                "max": args.max,
                "seed": args.seed,
                "n": args.n,
                "dtype": args.dtype,
            }
        else:  # story
            kwargs = {
                "model": args.model,
                "cfg": Config(window=args.window, sink=args.sink),
                "ctx": ctx,
                "task": [int(x) for x in args.task.split(",")],
                "frac": frac,
                "max": args.max,
                "seed": args.seed,
                "n": args.n,
                "dtype": args.dtype,
            }
        if args.out:
            kwargs["out"] = Path(args.out)
        fn(**kwargs)
        return 0

    kwargs = {
        "model_id": args.model,
        "window_size": args.window,
        "num_sinks": args.sinks,
    }
    # Map CLI flag names to function parameter names.
    flag_to_key = {
        "tasks": "tasks",
        "batch_size": "batch_size",
        "num_fewshot": "num_fewshot",
        "limit": "limit",
        "ctxs": "context_lens",
        "variants": "variants",
        "max_new": "max_new_tokens",
        "dtype": "dtype",
        "ctx_words": "context_word_lens",
        "qa_ids": "qa_ids",
    }
    for cli_name, fn_name in flag_to_key.items():
        if fn_name in keys and hasattr(args, cli_name):
            v = getattr(args, cli_name)
            if isinstance(v, str) and "," in v and fn_name in {
                "tasks", "variants", "qa_ids", "context_lens", "context_word_lens",
            }:
                v = [x.strip() for x in v.split(",") if x]
            kwargs[fn_name] = v
    if out:
        kwargs["output_path"] = Path(out)
    fn(**kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())