"""02 — Needle-in-a-Haystack with ``beacon.find``.

What it demonstrates
--------------------
Build one NIAH sample (V1: "best thing to do in San Francisco is to eat a
sandwich"), wrap it in a 256-word synthetic filler, ask the patched model
to complete the answer, and check whether "sandwich" appears in the
completion via ``beacon.sample.contains``. This is the smallest unit of
the S-NIAH benchmark (Table 3).

Install (one time)::

    pip install -e .

Run::

    python examples/02_niah.py
"""

from __future__ import annotations

from beacon import Config, load
from beacon.find import samples
from beacon.sample import contains


def main() -> None:
    cfg = Config(window=64, sink=4)
    model, tok = load("openbmb/MiniCPM5-1B", cfg)

    cell = samples(ctx=[256], variant="1", seed=0, n=1)
    s = cell[0]

    ids = tok(s.prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=32, do_sample=False)
    completion = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    hit = contains(s.answer, completion)
    print(f"prompt len (words): {len(s.prompt.split())}")
    print(f"expected answer: {s.answer!r}")
    print(f"completion: {completion!r}")
    print(f"contains(expected, completion): {hit}")


if __name__ == "__main__":
    main()