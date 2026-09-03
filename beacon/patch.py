"""Core SWA-with-sinks attention mask and HF model patch.

Implements SWA(w, s) as defined in Jolicoeur-Martineau et al. (2026):
attend to the previous ``window`` tokens *and* the first ``sink`` tokens.

The patch is model-agnostic: it works with any HuggingFace causal LM whose
attention uses a 4-D additive mask (Llama, Mistral, Qwen2, Phi-3, Gemma, ...).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    """Hyper-parameters of a SWA-with-sinks mask."""

    window: int = 64
    sink: int = 4

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError(f"window must be > 0, got {self.window}")
        if self.sink < 0:
            raise ValueError(f"sink must be >= 0, got {self.sink}")


def mask(
    seq_q: int,
    seq_k: int,
    *,
    window: int,
    sink: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a SWA-with-sinks additive attention mask.

    Parameters
    ----------
    seq_q, seq_k
        Query and key sequence lengths. ``seq_k >= seq_q``. When ``seq_q < seq_k``
        we are decoding: a single new query attends to ``seq_k`` past keys.
    window, sink
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

    cache_len = seq_k - seq_q
    q_idx = (cache_len + q).expand(seq_q, seq_k)
    k_idx = k.expand(seq_q, seq_k)
    valid = (k_idx < sink) | ((q_idx - k_idx >= 0) & (q_idx - k_idx < window))

    out = torch.zeros(seq_q, seq_k, device=device, dtype=dtype)
    out.masked_fill_(~valid, float("-inf"))
    return out.view(1, 1, seq_q, seq_k)


def make_update(cfg: Config):
    """Return a function suitable for monkey-patching ``PreTrainedModel._update_causal_mask``.

    The closure has the same signature as the upstream method but produces a
    SWA-with-sinks mask instead of a full causal mask.
    """

    def update(self, attention_mask, input_tensor, cache_position, past_key_values_length, *args, **kwargs):
        dtype = input_tensor.dtype
        device = input_tensor.device
        seq_q = input_tensor.shape[1]
        seq_k = seq_q + (past_key_values_length or 0)

        attn_mask = mask(
            seq_q,
            seq_k,
            window=cfg.window,
            sink=cfg.sink,
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

    return update


def patch(model: nn.Module, cfg: Config) -> nn.Module:
    """Replace ``model``'s attention mask with SWA(w, s).

    Forces eager attention so the 4-D additive mask is honoured, then monkey-patches
    the model's ``_update_causal_mask`` to emit our mask. Returns the same object.
    """
    model.config._attn_implementation = "eager"
    model.__class__._update_causal_mask = make_update(cfg)
    return model


__all__ = ["Config", "mask", "patch"]