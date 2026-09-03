"""Sanity check the SWA mask logic and the HF patch contract.

Runs without any external model download. Verifies:

* ``swa_mask`` shape / dtype / value semantics
* ``SWAConfig`` validation
* ``SWAPatchedModel`` patch idempotence

Run: ``pytest tests/ -q`` (after ``pip install -e ".[dev]"``) or
``python tests/test_patch.py`` for a no-deps smoke test.
"""

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from swa.patch import SWAConfig, swa_mask, patch_swa, SWAPatchedModel  # noqa: E402


def test_mask_shape_and_dtype():
    m = swa_mask(8, 8, window_size=4, num_sinks=2, dtype=torch.float32)
    assert m.shape == (1, 1, 8, 8)
    assert m.dtype == torch.float32


def test_prefill_attends_to_sinks_and_window():
    w, s = 4, 2
    m = swa_mask(6, 6, window_size=w, num_sinks=s)
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
    m = swa_mask(new_q, cache_len + new_q, window_size=4, num_sinks=2)
    valid = torch.isfinite(m).view(-1)
    # For a single new query at position cache_len, attend to keys in [0, s) U [cache_len-w+1, cache_len]
    expected = torch.zeros(cache_len + new_q, dtype=torch.bool)
    expected[:2] = True
    expected[3:7] = True
    assert torch.equal(valid, expected), f"got {valid.int()} want {expected.int()}"


def test_swa_config_validation():
    import pytest

    with pytest.raises(ValueError):
        SWAConfig(window_size=0)
    with pytest.raises(ValueError):
        SWAConfig(num_sinks=-1)


def test_patch_on_tiny_lm():
    """End-to-end: build a tiny Llama-like causal LM, patch it, run forward."""
    from transformers import AutoConfig, AutoModelForCausalLM

    # Tiny config so the test is fast and CPU-friendly.
    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=2,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()

    wrapped = patch_swa(model, window_size=4, num_sinks=2)
    assert isinstance(wrapped, SWAPatchedModel)
    assert wrapped._patch_applied

    ids = torch.randint(0, config.vocab_size, (1, 10))
    with torch.no_grad():
        out = wrapped(input_ids=ids)
    assert out.logits.shape == (1, 10, config.vocab_size)


def test_patch_generate_decodes():
    """Verify decoding produces logits (no shape errors from the mask hook)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM", num_hidden_layers=2)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    wrapped = patch_swa(model, window_size=4, num_sinks=2)

    ids = tok("hello world", return_tensors="pt").input_ids
    with torch.no_grad():
        out = wrapped.generate(ids, max_new_tokens=8, do_sample=False)
    assert out.shape[1] == ids.shape[1] + 8


def test_patch_with_gqa_like_minicpm5():
    """Verify SWA patch works on Llama GQA architecture (MiniCPM5 uses 16 Q / 2 KV heads)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=2,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    wrapped = patch_swa(model, window_size=4, num_sinks=2)

    ids = torch.randint(0, config.vocab_size, (1, 10))
    with torch.no_grad():
        # Prefill
        out = wrapped(input_ids=ids)
        assert out.logits.shape == (1, 10, config.vocab_size)
        # Decode one more token (single-key shape -> [B, 1, KV=11])
        out = wrapped(input_ids=ids[:, -1:])
        assert out.logits.shape == (1, 1, config.vocab_size)


if __name__ == "__main__":
    test_mask_shape_and_dtype()
    test_prefill_attends_to_sinks_and_window()
    test_decoding_mask_attends_to_recent_window()
    test_swa_config_validation()
    test_patch_on_tiny_lm()
    test_patch_generate_decodes()
    test_patch_with_gqa_like_minicpm5()
    print("All patch tests passed.")