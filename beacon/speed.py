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

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .patch import Config
from .seed import seed_all


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


def kv_swa(model, cfg: Config) -> float:
    """Steady-state KV-cache memory in MiB for SWA-with-sinks.

    Unlike full causal attention, a sliding window never needs to attend to the
    whole history: each layer only ever materializes K/V for the ``window``
    recent positions plus the ``sink`` tokens. So the steady-state memory is
    bounded and independent of ``seq_len``.

    ``2 (K+V) * num_layers * (window + sink) * hidden * dtype_bytes``.
    """
    params = model.parameters()
    bytes_per = torch.tensor([], dtype=next(params).dtype).element_size()
    size_bytes = (
        2
        * model.config.num_hidden_layers
        * (cfg.window + cfg.sink)
        * model.config.hidden_size
        * bytes_per
    )
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
    """Run one method across the ctx grid.

    Each ``ctx = c`` cell measures **steady-state decode**: the model prefills
    ``c`` tokens once (untimed) into a ``DynamicCache``, then the timed loop
    decodes ``step`` single tokens against that cache. No ``torch.cat`` grows
    inside the timed region, and ``kv_mib`` is reported at the prefill length.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache

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

    seed_all(seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)

    rows: list[Bench] = []
    for c in ctx:
        ids = torch.randint(
            low=0, high=model.config.vocab_size, size=(1, c), device=dev, dtype=torch.long, generator=gen
        )
        cache = DynamicCache()
        with torch.no_grad():
            _ = model(input_ids=ids, past_key_values=cache, use_cache=True)  # warmup prefill (not timed)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Steady-state decode: feed one new token per step against the prefill
        # cache; transformers appends it at the next absolute position. No
        # torch.cat grows in the timed region and the attention window stays
        # bounded for SWA.
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(step):
                next_tok = torch.randint(
                    low=0, high=model.config.vocab_size, size=(1, 1), device=dev, dtype=torch.long, generator=gen
                )
                _ = model(
                    input_ids=next_tok,
                    past_key_values=cache,
                    use_cache=True,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0
        tps = step / max(elapsed, 1e-9)
        latency_ms = (elapsed / step) * 1000.0
        kv_mib = kv_swa(model, cfg) if method == "swa" else kv(model, c)
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


__all__ = ["Bench", "Method", "METHOD", "kv", "kv_swa", "bench", "run"]