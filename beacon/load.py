"""Model and tokenizer loading.

Every benchmark needs the same three lines: ``AutoTokenizer.from_pretrained``,
``AutoModelForCausalLM.from_pretrained``, and a ``patch(model, cfg)`` call.
Keeping them here avoids that duplication across ``short``, ``find``,
``story`` and ``speed``.

Usage::

    from beacon import Config, load
    model, tok = load("openbmb/MiniCPM5-1B", Config(window=64, sink=4))
"""

from __future__ import annotations

import torch

from .patch import Config, patch


def load(
    model: str,
    cfg: Config | None = None,
    *,
    dtype: torch.dtype | str = torch.float16,
):
    """Load a HF causal LM and patch it with SWA.

    Parameters
    ----------
    model
        A HuggingFace model id (e.g. ``"openbmb/MiniCPM5-1B"``).
    cfg
        SWA hyperparameters. Defaults to ``Config()``.
    dtype
        ``torch.dtype`` or string name (``"float16"``, ``"bfloat16"``).

    Returns
    -------
    (model, tokenizer)
        The patched model and tokenizer are returned in eval mode. The
        tokenizer's pad token is set to eos if missing.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cfg is None:
        cfg = Config()
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype)
    m = patch(m, cfg)
    m.eval()
    return m, tok


__all__ = ["load"]