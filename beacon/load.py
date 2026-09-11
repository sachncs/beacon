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
from transformers import AutoModelForCausalLM, AutoTokenizer

from .patch import Config, patch

_DTYPE_ALIASES = {
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
}


def _coerce_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if not isinstance(dtype, str):
        raise TypeError(f"dtype must be a torch.dtype or string, got {type(dtype).__name__}")
    if dtype not in _DTYPE_ALIASES:
        raise ValueError(
            f"unknown dtype {dtype!r}; expected one of {sorted(_DTYPE_ALIASES)}"
        )
    return _DTYPE_ALIASES[dtype]


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
        ``torch.dtype`` or string name (``"float16"``, ``"bfloat16"``,
        ``"float32"``, ``"float64"``). Raises ``ValueError`` for unknown names.

    Returns
    -------
    (model, tokenizer)
        The patched model and tokenizer are returned in eval mode. The
        tokenizer's pad token is set to eos if missing.
    """
    if cfg is None:
        cfg = Config()
    dtype = _coerce_dtype(dtype)

    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype)
    m = patch(m, cfg)
    m.eval()
    return m, tok


__all__ = ["load"]
