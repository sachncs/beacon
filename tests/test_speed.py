"""Tests for beacon.speed — Method registry, kv() analytic formula."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from beacon.patch import Config
from beacon.speed import METHOD, Method, fa, kv, kv_swa, swa


def test_method_registry_has_fa_and_swa():
    assert set(METHOD.keys()) == {"fa", "swa"}
    assert all(isinstance(m, Method) for m in METHOD.values())


def test_fa_is_pass_through():
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    assert fa(m, None) is m


def test_swa_wraps_with_patch():
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    cfg = Config(window=8, sink=2)
    out = swa(m, cfg)
    assert out is m
    assert m.config._attn_implementation == "eager"


def test_kv_matches_analytic_formula():
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=2,
    )
    model = AutoModelForCausalLM.from_config(config)

    bytes_per = torch.tensor([], dtype=torch.float32).element_size()
    num_kv = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    expected = 2 * 2 * num_kv * head_dim * 128 * bytes_per / (1024 * 1024)
    assert abs(kv(model, 128) - expected) < 1e-9
    # Doubling the sequence length should double the KV-cache memory.
    assert abs(kv(model, 256) - 2 * expected) < 1e-9


def test_kv_with_gqa_uses_kv_heads_not_attention_heads():
    """For GQA the KV-cache cost grows with num_kv_heads * head_dim, not hidden_size."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=2,
    )
    model = AutoModelForCausalLM.from_config(config)
    bytes_per = torch.tensor([], dtype=torch.float32).element_size()
    naive = 2 * 2 * 128 * config.hidden_size * bytes_per / (1024 * 1024)
    correct = 2 * 2 * 2 * config.head_dim * 128 * bytes_per / (1024 * 1024)
    assert abs(kv(model, 128) - naive) > 1e-6
    assert abs(kv(model, 128) - correct) < 1e-9


def test_kv_swa_is_bounded_by_window_plus_sink():
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=2,
    )
    model = AutoModelForCausalLM.from_config(config)

    bytes_per = torch.tensor([], dtype=torch.float32).element_size()
    num_kv = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    cfg = Config(window=64, sink=4)
    expected = 2 * 2 * num_kv * head_dim * (64 + 4) * bytes_per / (1024 * 1024)
    assert abs(kv_swa(model, cfg) - expected) < 1e-9
    # SWA memory is independent of seq_len: a bigger window is the only driver.
    big = Config(window=256, sink=4)
    assert kv_swa(model, big) > kv_swa(model, cfg)
    assert abs(kv_swa(model, cfg) - kv_swa(model, Config(64, 4))) < 1e-12


def test_unknown_method_in_bench_raises():
    """bench() rejects method names not in METHOD."""
    from beacon.speed import bench

    with pytest.raises(ValueError):
        bench(
            model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
            method="bogus",
            cfg=Config(),
            ctx=[8],
            step=1,
        )


def test_bench_rejects_unknown_dtype_before_loading():
    """bench() validates dtype like every other runner; no getattr(torch, s)."""
    from beacon.speed import bench

    with pytest.raises(ValueError):
        bench(
            model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
            method="fa",
            cfg=Config(),
            ctx=[8],
            step=1,
            dtype="float81",  # not a torch dtype
        )


def test_bench_does_not_silently_substitute_model():
    """A load failure must propagate, never fall back to a different model."""
    from beacon.speed import bench

    # The bogus method check fires first, so use a valid method but an
    # unresolvable model id to force the real load path to fail.
    with pytest.raises(Exception):
        bench(
            model_id="definitely-not-a-real-model-id-xyz",
            method="fa",
            cfg=Config(),
            ctx=[8],
            step=1,
        )


if __name__ == "__main__":
    test_method_registry_has_fa_and_swa()
    test_fa_is_pass_through()
    test_swa_wraps_with_patch()
    test_kv_matches_analytic_formula()
    test_unknown_method_in_bench_raises()
    print("All speed tests passed.")