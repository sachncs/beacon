"""01 — Load a model, patch it, generate text.

What it demonstrates
--------------------
The minimal end-to-end use of beacon: ``load()`` returns a model whose
attention is the SWA-with-sinks mask (window=64, sink=4) instead of full
causal attention. The forward and ``.generate()`` paths use the patch
transparently; the model looks identical to a stock HuggingFace causal LM
from the caller's point of view.

Install (one time)::

    pip install -e .

Run::

    python examples/01_basic_patch.py

The script prints the patch stamp (``model.config._beacon``) so a reader
can verify the SWA configuration is active, then generates 32 tokens from
a short prompt and prints them. No GPU is required for the tiny model the
test suite uses; substitute ``openbmb/MiniCPM5-1B`` for a real run.
"""

from __future__ import annotations

from beacon import Config, load


def main() -> None:
    cfg = Config(window=64, sink=4)
    print(f"loading model with cfg window={cfg.window} sink={cfg.sink}")
    model, tok = load("openbmb/MiniCPM5-1B", cfg)

    beacon = getattr(model.config, "_beacon", None)
    print(f"model.config._attn_implementation = {model.config._attn_implementation}")
    print(f"model.config._beacon = {beacon}")

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Name one fruit in one word."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)

    out = model.generate(ids, max_new_tokens=32, do_sample=False)
    completion = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"completion: {completion!r}")


if __name__ == "__main__":
    main()