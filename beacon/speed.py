"""Speed and memory benchmark from Figure 2.

Measures:

* **Throughput** in tokens/sec at decoding time, for context lengths
  ``128..256K`` and batch size 1.
* **KV-cache memory** (MiB) as a function of context length.

We compare:

* **FA**  — full causal attention (eager backend, our reference).
* **SWA(w, s)** — Sliding Window Attention with ``w`` previous tokens and
  ``s`` sink tokens.

Linear attention and Linear+SWA baselines require custom CUDA kernels
(ThunderKittens in the paper); reproducing those is out of scope here.
We expose the same measurement harness so users with custom kernels can
plug in their own ``forward`` callables.

Hardware target (per the paper): RTX PRO 6000 Blackwell Max-Q, 4-layer
Transformer with embed=1024, 16 heads of dim 64, fp16, batch 1.
Falls back gracefully to any local GPU/CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from .patch import Config, patch


@dataclass
class BenchResult:
    method: str
    context_len: int
    tokens_per_sec: float
    kv_cache_mib: float
    decode_latency_ms: float


def _build_model(model_id: str, dtype: torch.dtype, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    return model, tok


def _kv_cache_mib(model) -> float:
    """Approximate KV cache memory: 2 * num_layers * seq_len * hidden * dtype_bytes."""
    cfg = model.config
    seq_len = getattr(model, "_current_seq_len", 0)
    dtype_bytes = torch.tensor([], dtype=next(model.parameters()).dtype).element_size()
    # Llama/Mistral/Qwen style: 2 (K+V) * num_hidden_layers * seq * hidden
    size_bytes = 2 * cfg.num_hidden_layers * seq_len * cfg.hidden_size * dtype_bytes
    return size_bytes / (1024 * 1024)


def _benchmark(
    *,
    model_id: str,
    method: str,
    window_size: int,
    num_sinks: int,
    context_lens: list[int],
    decode_steps: int = 64,
    dtype: str = "float16",
    device: str | None = None,
) -> list[BenchResult]:
    """Run a benchmark for one ``method`` across ``context_lens``."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Use a tiny random model of the paper's specs if available; otherwise
    # use the requested model. The paper specifies a synthetic config, but
    # for a usable benchmark we use the user's chosen model and let the
    # window-size relative behaviour be what matters.
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_id,
            num_hidden_layers=4,
            hidden_size=1024,
            intermediate_size=4096,
            num_attention_heads=16,
        )
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype).to(dev)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(dev)

    model.eval()

    if method == "swa":
        model = patch(model, Config(window=window_size, sink=num_sinks))
    elif method == "fa":
        pass  # default
    else:
        raise ValueError(f"unknown method: {method}")

    results: list[BenchResult] = []

    for ctx in context_lens:
        ids = torch.randint(
            low=0,
            high=model.config.vocab_size,
            size=(1, ctx),
            device=dev,
            dtype=torch.long,
        )
        with torch.no_grad():
            # Warmup prefill (does not count toward decode timing).
            _ = model(input_ids=ids)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Decode ``decode_steps`` tokens.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(decode_steps):
                # Greedy one-token generation; we feed back the last token.
                out = model(input_ids=ids[:, -1:])
                next_id = out.logits[:, -1].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_id], dim=1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tps = decode_steps / max(elapsed, 1e-9)
        kv_mib = _kv_cache_mib(model)
        latency_ms = (elapsed / decode_steps) * 1000.0

        results.append(BenchResult(method, ctx, tps, kv_mib, latency_ms))
        print(f"{method} ctx={ctx:>6d}  tps={tps:7.1f}  latency={latency_ms:6.2f}ms  KV={kv_mib:7.2f} MiB")

    return results


def run_speed_mem(
    *,
    model_id: str,
    window_size: int = 64,
    num_sinks: int = 4,
    context_lens: list[int] | None = None,
    decode_steps: int = 64,
    methods: list[str] | None = None,
    dtype: str = "float16",
    out: Path | None = None,
) -> dict[str, list[BenchResult]]:
    if context_lens is None:
        context_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 131072, 262144]
    if methods is None:
        methods = ["fa", "swa"]

    all_results: dict[str, list[BenchResult]] = {}
    for m in methods:
        print(f"=== {m} ===")
        all_results[m] = _benchmark(
            model_id=model_id,
            method=m,
            window_size=window_size,
            num_sinks=num_sinks,
            context_lens=context_lens,
            decode_steps=decode_steps,
            dtype=dtype,
        )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {m: [asdict(r) for r in rs] for m, rs in all_results.items()},
            indent=2,
        ))
    return all_results


def _cli() -> None:
    p = argparse.ArgumentParser(description="Speed/memory benchmark (Figure 2)")
    p.add_argument("--model", required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--sinks", type=int, default=4)
    p.add_argument("--ctxs", default="128,256,512,1024,2048,4096,8192,16384,65536")
    p.add_argument("--decode-steps", type=int, default=64)
    p.add_argument("--methods", default="fa,swa")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    ctxs = [int(x) for x in args.ctxs.split(",")]
    methods = [x.strip() for x in args.methods.split(",") if x]

    run_speed_mem(
        model_id=args.model,
        window_size=args.window,
        num_sinks=args.sinks,
        context_lens=ctxs,
        decode_steps=args.decode_steps,
        methods=methods,
        dtype=args.dtype,
        out=args.out,
    )


if __name__ == "__main__":
    _cli()