"""Speed and memory benchmark from Figure 2.

Measures, for each (method, ctx) cell:

* **Throughput** (tokens/sec) at decoding time, batch size 1.
* **KV-cache memory** (MiB).
* **Per-token decode latency** (ms).

Methods are strategies in :data:`METHOD`. Currently:

* ``fa``  — full causal attention (eager backend, our reference).
* ``swa`` — Sliding Window Attention with ``window`` previous tokens and
  ``sink`` sink tokens.

Linear attention and Linear+SWA baselines require custom CUDA kernels
(ThunderKittens in the paper); reproducing those is out of scope here.
Add a new ``Method`` to ``METHOD`` to plug in a custom kernel.

Hardware target (per the paper): RTX PRO 6000 Blackwell Max-Q, 4-layer
Transformer with embed=1024, 16 heads of dim 64, fp16, batch 1.
Falls back gracefully to any local GPU/CPU.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .patch import Config


@dataclass(frozen=True)
class Bench:
    """One row of benchmark output."""

    method: str
    ctx: int
    tps: float
    kv_mib: float
    latency: float


@dataclass(frozen=True)
class Method:
    """An attention strategy: a name plus a callable that wraps a model."""

    name: str
    apply: callable


def fa(model, _cfg) -> torch.nn.Module:
    """Full causal attention — pass-through."""
    return model


def swa(model, cfg: Config) -> torch.nn.Module:
    """Sliding Window Attention with sinks."""
    from .patch import patch
    return patch(model, cfg)


METHOD: dict[str, Method] = {
    "fa": Method("fa", fa),
    "swa": Method("swa", swa),
}


def kv(model, seq_len: int) -> float:
    """KV-cache memory in MiB for a Llama-style model at ``seq_len`` tokens.

    ``2 (K+V) * num_layers * seq_len * hidden * dtype_bytes``.
    """
    cfg = model.config
    bytes_per = torch.tensor([], dtype=next(model.parameters()).dtype).element_size()
    size_bytes = 2 * cfg.num_hidden_layers * seq_len * cfg.hidden_size * bytes_per
    return size_bytes / (1024 * 1024)


def bench(
    *,
    model_id: str,
    method: str,
    cfg: Config,
    ctx: list[int],
    step: int = 64,
    dtype: str = "float16",
    device: str | None = None,
    seed: int = 0,
) -> list[Bench]:
    """Run one method across the ctx grid."""
    from transformers import AutoConfig, AutoModelForCausalLM

    if method not in METHOD:
        raise ValueError(f"unknown method: {method!r}; choose from {list(METHOD)}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Use a tiny random model of the paper's specs if available; otherwise
    # fall back to the requested model. The relative FA vs SWA behaviour
    # is what matters for the benchmark.
    try:
        config = AutoConfig.from_pretrained(
            model_id,
            num_hidden_layers=4,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
        )
        model = AutoModelForCausalLM.from_config(config, torch_dtype=getattr(torch, dtype)).to(dev)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, dtype)).to(dev)

    model.eval()
    model = METHOD[method].apply(model, cfg)

    rows: list[Bench] = []
    torch.manual_seed(seed)
    for c in ctx:
        ids = torch.randint(low=0, high=model.config.vocab_size, size=(1, c), device=dev, dtype=torch.long)
        with torch.no_grad():
            _ = model(input_ids=ids)  # warmup prefill (not timed)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(step):
                out = model(input_ids=ids[:, -1:])
                next_id = out.logits[:, -1].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_id], dim=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0
        tps = step / max(elapsed, 1e-9)
        latency_ms = (elapsed / step) * 1000.0
        kv_mib = kv(model, ids.shape[1])
        rows.append(Bench(method=method, ctx=c, tps=tps, kv_mib=kv_mib, latency=latency_ms))
        print(f"{method} ctx={c:>6d}  tps={tps:7.1f}  latency={latency_ms:6.2f}ms  KV={kv_mib:7.2f} MiB")
    return rows


def run(
    *,
    model: str,
    cfg: Config,
    ctx: list[int] | None = None,
    method: tuple[str, ...] | list[str] = ("fa", "swa"),
    step: int = 64,
    seed: int = 0,
    dtype: str = "float16",
    out: Path | None = None,
) -> dict[str, list[Bench]]:
    """Run all configured methods across the ctx grid."""
    if ctx is None:
        ctx = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 131072, 262144]

    out_dict: dict[str, list[Bench]] = {}
    for m in method:
        print(f"=== {m} ===")
        out_dict[m] = bench(
            model_id=model,
            method=m,
            cfg=cfg,
            ctx=ctx,
            step=step,
            dtype=dtype,
            seed=seed,
        )

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({m: [asdict(r) for r in rs] for m, rs in out_dict.items()}, indent=2))
    return out_dict


def cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Speed / KV-cache memory benchmark (Figure 2)")
    p.add_argument("--model", required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--sink", type=int, default=4)
    p.add_argument("--ctx", default="128,256,512,1024,2048,4096,8192,16384,65536")
    p.add_argument("--method", default="fa,swa")
    p.add_argument("--step", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    run(
        model=args.model,
        cfg=Config(window=args.window, sink=args.sink),
        ctx=[int(x) for x in args.ctx.split(",")],
        method=tuple(x.strip() for x in args.method.split(",") if x),
        step=args.step,
        seed=args.seed,
        dtype=args.dtype,
        out=args.out,
    )
    return 0


__all__ = ["Bench", "Method", "METHOD", "kv", "bench", "run", "cli"]


if __name__ == "__main__":
    raise SystemExit(cli())