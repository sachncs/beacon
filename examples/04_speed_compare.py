"""04 — Speed / KV-cache memory benchmark (Figure 2, single method).

What it demonstrates
--------------------
Time steady-state decode for the SWA-with-sinks method at a small ctx grid
and report both the prefill KV cost (``kv_prefill_mib``) and the
steady-state cost (``kv_mib``). The two columns diverge as ctx grows:
steady-state KV is bounded by ``window + sink``, prefill KV grows linearly.

The full ``beacon.speed.run`` loop sweeps multiple methods; this script
shows the single-method variant and is the smallest unit of Figure 2.

Install (one time)::

    pip install -e .

Run::

    python examples/04_speed_compare.py
"""

from __future__ import annotations

from beacon import Config
from beacon.speed import bench


def main() -> None:
    cfg = Config(window=64, sink=4)
    rows = bench(
        model_id="openbmb/MiniCPM5-1B",
        method="swa",
        cfg=cfg,
        ctx=[128, 512, 2048],
        step=16,
        dtype="float16",
    )
    print()
    print(f"{'ctx':>6}  {'tps':>6}  {'latency_ms':>10}  {'KV(prefill)':>11}  {'KV(steady)':>10}")
    for r in rows:
        print(
            f"{r.ctx:>6}  {r.tps:>6.1f}  {r.latency:>10.2f}  "
            f"{r.kv_prefill_mib:>9.2f}MiB  {r.kv_mib:>8.2f}MiB"
        )


if __name__ == "__main__":
    main()