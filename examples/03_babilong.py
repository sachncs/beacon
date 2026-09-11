"""03 — Single QA1 story from the BABILong benchmark.

What it demonstrates
--------------------
Build a QA1 ("single supporting fact") sample with ``beacon.story.samples``,
ask the patched model the question, and report whether the location name
appears in the greedy completion. This is one cell of Table 4; ``run``
typically sweeps over (task, ctx, frac).

Install (one time)::

    pip install -e .

Run::

    python examples/03_babilong.py
"""

from __future__ import annotations

from beacon import Config, load
from beacon.sample import contains
from beacon.story import samples


def main() -> None:
    cfg = Config(window=64, sink=4)
    model, tok = load("openbmb/MiniCPM5-1B", cfg)

    cell = samples(task=[1], ctx=[512], frac=[0.5], seed=0, n=1)
    s = cell[0]

    ids = tok(s.prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=32, do_sample=False)
    completion = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    hit = contains(s.answer, completion)
    print(f"task: QA{s.meta['task']}  ctx (words): {s.meta['ctx']}")
    print(f"expected answer: {s.answer!r}")
    print(f"completion: {completion!r}")
    print(f"contains(expected, completion): {hit}")


if __name__ == "__main__":
    main()