"""Core SWA-with-sinks attention mask and HF model patch.

Implements SWA(w, s) as defined in Jolicoeur-Martineau et al. (2026):
attend to the previous ``w`` tokens *and* the first ``s`` tokens (sinks).

The patch is model-agnostic: it works with any HuggingFace causal LM whose
attention uses a 4-D additive mask (Llama, Mistral, Qwen2, Phi-3, Gemma, ...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class SWAConfig:
    """Hyper-parameters of a SWA-with-sinks mask."""

    window_size: int = 64
    num_sinks: int = 4

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")
        if self.num_sinks < 0:
            raise ValueError(f"num_sinks must be >= 0, got {self.num_sinks}")


def swa_mask(
    seq_q: int,
    seq_k: int,
    *,
    window_size: int,
    num_sinks: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a SWA-with-sinks additive attention mask.

    Parameters
    ----------
    seq_q, seq_k
        Query and key sequence lengths. ``seq_k >= seq_q``. When ``seq_q < seq_k``
        we are decoding: a single new query attends to ``seq_k`` past keys.
    window_size, num_sinks
        SWA hyperparameters (w, s).
    device, dtype
        Output device/dtype. Defaults are CPU + float32; callers typically cast.

    Returns
    -------
    torch.Tensor
        A ``(1, 1, seq_q, seq_k)`` additive mask. ``0`` = attend, ``-inf`` = mask out.
    """
    if seq_k < seq_q:
        raise ValueError(f"seq_k ({seq_k}) must be >= seq_q ({seq_q})")

    q = torch.arange(seq_q, device=device).view(-1, 1)
    k = torch.arange(seq_k, device=device).view(1, -1)

    if seq_q == seq_k:
        q_idx = q.expand(seq_q, seq_k)
        k_idx = k.expand(seq_q, seq_k)
        valid = (k_idx < num_sinks) | ((q_idx - k_idx >= 0) & (q_idx - k_idx < window_size))
    else:
        cache_len = seq_k - seq_q
        q_idx = (cache_len + q).expand(seq_q, seq_k)
        k_idx = k.expand(seq_q, seq_k)
        valid = (k_idx < num_sinks) | ((q_idx - k_idx >= 0) & (q_idx - k_idx < window_size))

    mask = torch.zeros(seq_q, seq_k, device=device, dtype=dtype)
    mask.masked_fill_(~valid, float("-inf"))
    return mask.view(1, 1, seq_q, seq_k)


def _make_swa_update_causal_mask(swa: SWAConfig):
    """Return a function suitable for monkey-patching ``PreTrainedModel._update_causal_mask``.

    The returned function has the same signature as the upstream method but produces
    a SWA-with-sinks mask instead of a full causal mask.
    """

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values_length,
        *args,
        **kwargs,
    ):
        # Always start from a causal-future mask: never let query i attend to key j > i.
        # Then carve out the SWA window on top of that.
        dtype = input_tensor.dtype
        device = input_tensor.device
        seq_q = input_tensor.shape[1]
        seq_k = seq_q + (past_key_values_length or 0)

        attn_mask = swa_mask(
            seq_q,
            seq_k,
            window_size=swa.window_size,
            num_sinks=swa.num_sinks,
            device=device,
            dtype=torch.float32,
        ).to(dtype)

        if attention_mask is not None:
            # ``attention_mask`` from the tokenizer is a (B, K) padding mask of 0/1.
            # Convert to additive form and broadcast against our 4-D mask.
            pad_mask = attention_mask
            if pad_mask.dim() == 2:
                pad_mask = pad_mask[:, None, None, :].to(dtype)
                pad_mask = (1.0 - pad_mask) * torch.finfo(dtype).min
            attn_mask = attn_mask + pad_mask

        return attn_mask

    return _update_causal_mask


class SWAPatchedModel(nn.Module):
    """Wrapper that lazily swaps in the SWA mask on a loaded causal LM.

    The wrapper holds a reference to the underlying HF model and exposes the
    standard ``generate`` / ``forward`` interface. No weights are modified.
    """

    def __init__(self, model: nn.Module, swa: SWAConfig):
        super().__init__()
        self.model = model
        self.swa = swa
        self._patch_applied = False
        self._apply_patch()

    def _apply_patch(self) -> None:
        if self._patch_applied:
            return
        # Force eager attention so our 4-D additive mask is actually used.
        # SDPA / Flash kernels may or may not honour additive masks, depending on version.
        if hasattr(self.model.config, "_attn_implementation"):
            self.model.config._attn_implementation = "eager"
        if hasattr(self.model, "config"):
            self.model.config._attn_implementation = "eager"
        # Monkey-patch the model's mask updater.
        fn = _make_swa_update_causal_mask(self.swa)
        self.model.__class__._update_causal_mask = fn
        self._patch_applied = True

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def __getattr__(self, name: str):
        # Delegate everything else (parameters, config, ...) to the wrapped model.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def patch_swa(model: nn.Module, *, window_size: int = 64, num_sinks: int = 4) -> SWAPatchedModel:
    """Return a wrapper around ``model`` that uses SWA(w, s) attention.

    Parameters
    ----------
    model
        A loaded HuggingFace causal LM.
    window_size
        ``w``: how many previous tokens each query attends to.
    num_sinks
        ``s``: how many leading tokens are always attended to.

    Returns
    -------
    SWAPatchedModel
        Thin wrapper; drop-in replacement for the original module.
    """
    swa = SWAConfig(window_size=window_size, num_sinks=num_sinks)
    return SWAPatchedModel(model, swa)


__all__ = [
    "SWAConfig",
    "swa_mask",
    "patch_swa",
    "SWAPatchedModel",
]