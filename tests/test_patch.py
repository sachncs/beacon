"""Sanity check the SWA mask logic and the HF patch contract.

Runs without any external model download. Verifies:

* ``mask`` shape / dtype / value semantics
* ``Config`` validation
* ``patch`` returns the same model, sets eager attention, and produces correct
  prefill + decode shapes on tiny Llama + GQA Llama (MiniCPM5 layout).

Run: ``pytest tests/ -q`` (after ``pip install -e ".[dev]"``) or
``python tests/test_patch.py`` for a no-deps smoke test.
"""

import importlib

import pytest
import torch

from beacon.patch import Config, mask, patch


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
    with pytest.raises(ValueError):
        mask(seq_q=4, seq_k=2, window=4, sink=2)


def test_config_validation():
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


def test_architecture_module_failure_raises_not_implemented():
    """A model class that cannot be resolved to an importable module errors loudly."""
    from beacon.patch import architecture_module

    class Fake:
        pass

    Fake.__module__ = "definitely.not.a.real.beacon.architecture.module"
    with pytest.raises(NotImplementedError):
        architecture_module(Fake())


def attended_mask(model, ids, **forward_kwargs):
    """The ``(B, 1, Q, KV)`` additive attention mask ``model`` used during forward.

    ``patch`` installs the SWA builder as the ``create_causal_mask`` of the
    model's own architecture module (``beacon.patch.swa_create_causal_mask``), so
    this wraps that installed builder to capture its output while the model runs,
    then restores it. ``0.0`` marks an attended token; the eager backend masks
    non-attended tokens with ``torch.finfo(dtype).min`` (a finite, very-negative
    value), so callers compare to ``0.0``.
    """
    import warnings

    from beacon.patch import architecture_module

    module = architecture_module(model)
    installed = module.create_causal_mask
    captured = {}

    def wrapping(*args, **kwargs):
        mask_tensor = installed(*args, **kwargs)
        if mask_tensor is not None:
            captured["mask"] = mask_tensor
        return mask_tensor

    module.create_causal_mask = wrapping
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                model(input_ids=ids, **forward_kwargs)
    finally:
        module.create_causal_mask = installed
    return captured["mask"]


def test_patch_applies_swa_mask_not_full_causal():
    """REGRESSION (flagship): the patch must change the model's actual attention mask.

    Before the create_causal_mask rewrite this test failed: the old patch set
    ``_update_causal_mask``, which no modern transformers calls, so the model kept
    full causal attention. This asserts the model really attends with
    sinks + sliding window, not full causal.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=1,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    window, sink = 4, 2
    patch(model, Config(window=window, sink=sink))

    seq = 16
    ids = torch.randint(0, config.vocab_size, (1, seq))
    amask = attended_mask(model, ids)
    assert amask.shape == (1, 1, seq, seq)

    attend = amask[0, 0] == 0.0
    last = seq - 1
    # full causal would give all `seq` entries; SWA(w,s) gives sink + window.
    assert int(attend[last].sum()) == sink + window, attend[last].int().tolist()
    # every row attends to all sink tokens.
    assert bool(attend[:, :sink].all())
    # no row attends to a future token outside the sink region (sinks are global).
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    future_non_sink = future.clone()
    future_non_sink[:, :sink] = False
    assert not bool(attend[future_non_sink].any())


def test_mask_helper_agrees_with_patched_model():
    """mask() and the patched model's eager create_causal_mask are one predicate.

    Two implementations of the SWA-with-sinks rule (the standalone tensor helper
    mask() and the model's create_causal_mask factory) must produce the exact
    same attend-set, or tests written against mask() could silently miss a model
    that behaves differently. This cross-checks several (window, sink) configs.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=1,
    )
    seq = 16
    ids = torch.randint(0, config.vocab_size, (1, seq))

    for window, sink in [(4, 2), (8, 1), (2, 4), (16, 0)]:
        model = AutoModelForCausalLM.from_config(config).eval()
        patch(model, Config(window=window, sink=sink))
        amask = attended_mask(model, ids)

        expected = mask(seq, seq, window=window, sink=sink, dtype=amask.dtype)
        # eager backend blocks with torch.finfo(dtype).min, not -inf; the sentinel only matters via (== 0.0).
        assert torch.equal(amask[0, 0] == 0.0, expected[0, 0] == 0.0)


def test_patch_cross_instance_independence():
    """Two patched models of the same architecture use their own Config, not the last patch."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=1,
    )

    def fresh_model():
        c = AutoConfig.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM", num_hidden_layers=1
        )
        return AutoModelForCausalLM.from_config(c).eval()

    m1, m2 = fresh_model(), fresh_model()
    patch(m1, Config(window=4, sink=2))
    patch(m2, Config(window=8, sink=1))

    seq = 12
    ids = torch.randint(0, config.vocab_size, (1, seq))
    a1 = attended_mask(m1, ids)[0, 0, seq - 1] == 0.0
    a2 = attended_mask(m2, ids)[0, 0, seq - 1] == 0.0
    assert int(a1.sum()) == 2 + 4  # sink=2, window=4
    assert int(a2.sum()) == 1 + 8  # sink=1, window=8


def test_patch_decode_mask_uses_window_relative_to_cache():
    """During decode the last query attends to sinks + the trailing window of keys."""
    from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache

    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=1,
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    window, sink = 4, 2
    patch(model, Config(window=window, sink=sink))

    prefill = torch.randint(0, config.vocab_size, (1, 10))
    cache = DynamicCache()
    with torch.no_grad():
        model(input_ids=prefill, past_key_values=cache, use_cache=True)
    amask = attended_mask(model, prefill[:, -1:], past_key_values=cache, use_cache=True)
    # one new query over the 10 cached keys.
    assert amask.shape == (1, 1, 1, 11)
    attend = amask[0, 0, 0] == 0.0
    assert int(attend.sum()) == sink + window


def test_mask_edge_cases():
    w, s = 3, 1
    assert mask(0, 0, window=w, sink=s).shape == (1, 1, 0, 0)  # empty prefill
    # window >= seq_k degenerates to causal union the global sink columns.
    m = mask(5, 5, window=10, sink=2)
    q = torch.arange(5).view(-1, 1)
    k = torch.arange(5).view(1, -1)
    expect = (k <= q) | (k < 2)
    assert torch.equal(torch.isfinite(m).view(5, 5), expect)
    # sink == seq_k covers everything.
    m = mask(3, 3, window=1, sink=3)
    assert torch.equal(torch.isfinite(m).view(3, 3), torch.ones(3, 3, dtype=torch.bool))
    # sink == 0 and window == 1 -> only the diagonal is attended.
    m = mask(4, 4, window=1, sink=0)
    expect = torch.eye(4, dtype=torch.bool)
    assert torch.equal(torch.isfinite(m).view(4, 4), expect)


def test_mask_sink_overflow_edge():
    # sink larger than the key length is safe (masks everything a query can reach).
    m = mask(2, 2, window=1, sink=5)
    assert torch.equal(torch.isfinite(m).view(2, 2), torch.ones(2, 2, dtype=torch.bool))


def test_patch_forces_eager_and_stamps_config():
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=1,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    patch(model, Config(window=8, sink=3))
    assert model.config._attn_implementation == "eager"
    beacon = model.config._beacon
    assert beacon["window"] == 8
    assert beacon["sink"] == 3
    assert beacon["applied"] is True
    assert "previous_attn_implementation" in beacon


def test_patch_preserves_padding():
    """A padded batch still gets the correct SWA mask: sinks + window per position."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=1,
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    window, sink = 4, 2
    patch(model, Config(window=window, sink=sink))

    seq = 12
    ids = torch.randint(1, config.vocab_size, (2, seq))
    attn_mask = torch.ones(2, seq, dtype=torch.long)
    amask = attended_mask(model, ids, attention_mask=attn_mask)
    attend = amask[0, 0] == 0.0  # batch row 0 is un-padded
    assert int(attend[seq - 1].sum()) == sink + window

def test_patch_unsupported_encoder_model():
    """An encoder model that does not route through create_causal_mask raises NotImplementedError."""
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-BertForMaskedLM",
        num_hidden_layers=1,
    )
    model = AutoModel.from_config(config)
    with pytest.raises(NotImplementedError):
        patch(model, Config(window=4, sink=2))


def test_patch_old_transformers_raises(monkeypatch):
    """patch() refuses to run on an unsupported (pre-create_causal_mask) transformers version."""
    patch_mod = importlib.import_module("beacon.patch")
    monkeypatch.setattr(patch_mod, "transformers_version", lambda: (4, 43))
    with pytest.raises(RuntimeError):
        patch_mod.patch(None, Config(window=4, sink=2))


def test_transformers_version_parses_pep440(monkeypatch):
    """The version parser must handle PEP 440 suffixes without truncation."""
    import importlib

    import transformers as _tf

    patch_mod = importlib.import_module("beacon.patch")

    for raw, expected in [
        ("4.55.0", (4, 55)),
        ("4.55.0rc1", (4, 55)),
        ("4.55.0.post1", (4, 55)),
        ("4.55.0.dev0", (4, 55)),
        ("5.0", (5, 0)),
        ("4.56", (4, 56)),
    ]:
        monkeypatch.setattr(_tf, "__version__", raw)
        assert patch_mod.transformers_version() == expected, raw


def test_transformers_version_rejects_garbage(monkeypatch):
    import importlib

    import transformers as _tf

    patch_mod = importlib.import_module("beacon.patch")

    for bad in ("", "abc", "1", "1.dev"):
        monkeypatch.setattr(_tf, "__version__", bad)
        with pytest.raises(RuntimeError):
            patch_mod.transformers_version()


def test_unpatch_restores_upstream_and_clears_stamp():
    """unpatch() restores the upstream create_causal_mask and clears config._beacon."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from beacon.patch import unpatch

    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=1,
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    patch(model, Config(window=4, sink=2))
    assert getattr(model.config, "_beacon", None) is not None

    unpatch(model)
    assert not hasattr(model.config, "_beacon")


def test_unpatch_is_noop_on_unpatched_model():
    """unpatch() on a fresh model leaves it untouched."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from beacon.patch import unpatch

    config = AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        num_hidden_layers=1,
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    out = unpatch(model)
    assert out is model
    assert not hasattr(model.config, "_beacon")


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
