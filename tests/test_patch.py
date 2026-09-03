"""Sanity check the SWA mask logic and the HF patch contract.

Runs without any external model download. Verifies:

* ``mask`` shape / dtype / value semantics
* ``Config`` validation
* ``patch`` returns the same model, sets eager attention, and produces correct
  prefill + decode shapes on tiny Llama + GQA Llama (MiniCPM5 layout).

Run: ``pytest tests/ -q`` (after ``pip install -e ".[dev]"``) or
``python tests/test_patch.py`` for a no-deps smoke test.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from beacon.patch import Config, mask, patch  # noqa: E402


def test_mask_shape_and_dtype():
    m = mask(8, 8, window=4, sink=2, dtype=torch.float32)
    assert m.shape == (1, 1, 8, 8)
    assert m.dtype == torch.float32


def test_prefill_attends_to_sinks_and_window():
    w, s = 4, 2
    m = mask(6, 6, window=w, sink=s)
    finite = torch.isfinite(m).view(6, 6)
    # row i attends to keys in [0, s) UNION (i-w, i]
    for i in range(6):
        window = torch.zeros(6, dtype=torch.bool)
        lo, hi = max(0, i - w + 1), i + 1
        window[lo:hi] = True
        sinks = torch.zeros(6, dtype=torch.bool)
        sinks[:s] = True
        expected = window | sinks
        assert torch.equal(finite[i], expected), f"row {i}: got {finite[i].int()} want {expected.int()}"


def test_decoding_mask_attends_to_recent_window():
    cache_len, new_q = 6, 1
    m = mask(new_q, cache_len + new_q, window=4, sink=2)
    valid = torch.isfinite(m).view(-1)
    # For a single new query at position cache_len, attend to keys in [0, s) U [cache_len-w+1, cache_len]
    expected = torch.zeros(cache_len + new_q, dtype=torch.bool)
    expected[:2] = True
    expected[3:7] = True
    assert torch.equal(valid, expected), f"got {valid.int()} want {expected.int()}"


def test_mask_rejects_seq_k_lt_seq_q():
    import pytest

    with pytest.raises(ValueError):
        mask(seq_q=4, seq_k=2, window=4, sink=2)


def test_config_validation():
    import pytest

    with pytest.raises(ValueError):
        Config(window=0)
    with pytest.raises(ValueError):
        Config(sink=-1)


def test_patch_returns_same_model():
    """patch() must return the model it was called on (no wrapper)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=2,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    out = patch(model, Config(window=4, sink=2))
    assert out is model
    assert model.config._attn_implementation == "eager"


def test_patch_on_tiny_lm():
    """End-to-end: build a tiny Llama-like causal LM, patch it, run forward."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=2,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    model = patch(model, Config(window=4, sink=2))

    ids = torch.randint(0, config.vocab_size, (1, 10))
    with torch.no_grad():
        out = model(input_ids=ids)
    assert out.logits.shape == (1, 10, config.vocab_size)


def test_patch_generate_decodes():
    """Verify decoding produces logits (no shape errors from the mask hook)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM", num_hidden_layers=2)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    model = patch(model, Config(window=4, sink=2))

    ids = tok("hello world", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=8, do_sample=False)
    assert out.shape[1] == ids.shape[1] + 8


def test_patch_with_gqa_like_minicpm5():
    """Verify SWA patch works on Llama GQA architecture (MiniCPM5 uses 16 Q / 2 KV heads)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=2,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    model = patch(model, Config(window=4, sink=2))

    ids = torch.randint(0, config.vocab_size, (1, 10))
    with torch.no_grad():
        # Prefill
        out = model(input_ids=ids)
        assert out.logits.shape == (1, 10, config.vocab_size)
        # Decode one more token (single-key shape -> [B, 1, KV=11])
        out = model(input_ids=ids[:, -1:])
        assert out.logits.shape == (1, 1, config.vocab_size)


def test_patch_is_idempotent():
    """Calling patch() twice with the same Config must not break the model."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=2,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    model = patch(model, Config(window=4, sink=2))
    model = patch(model, Config(window=4, sink=2))

    ids = torch.randint(0, config.vocab_size, (1, 4))
    with torch.no_grad():
        out = model(input_ids=ids)
    assert out.logits.shape == (1, 4, config.vocab_size)


if __name__ == "__main__":
    test_mask_shape_and_dtype()
    test_prefill_attends_to_sinks_and_window()
    test_decoding_mask_attends_to_recent_window()
    test_mask_rejects_seq_k_lt_seq_q()
    test_config_validation()
    test_patch_returns_same_model()
    test_patch_on_tiny_lm()
    test_patch_generate_decodes()
    test_patch_with_gqa_like_minicpm5()
    test_patch_is_idempotent()
    print("All patch tests passed.")