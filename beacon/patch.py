"""Sliding Window Attention with sinks (SWA-with-sinks).

Implements ``SWA(w, s)`` as used in the reproduction: a query at absolute
position ``q`` attends to keys in ``[0, sink)`` (the sink tokens) **and** to the
previous ``window`` keys ending at ``q``, i.e. keys ``[q - window + 1, q]``.
``window`` counts the current position; it is the analogue of a sliding window
of length ``window`` with causal (no-future) masking.

The patch is applied to a HuggingFace decoder LM whose attention is driven by
the additive eager mask built in ``transformers.masking_utils.create_causal_mask``
(the shared free function used by Llama, Mistral, Qwen, Gemma, Phi, GPT-2 and
most other decoder families). ``patch`` replaces that function on the model's
own architecture module with a per-model builder so the model's *forward* pass
builds the SWA-with-sinks mask instead of a full causal mask.
"""

from __future__ import annotations

import importlib
import functools
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

MIN_TRANSFORMERS = (4, 55)

__all__ = ["Config", "mask", "swa_mask_function", "swa_create_causal_mask", "patch"]


@dataclass
class Config:
    """Hyper-parameters of an SWA-with-sinks mask.

    ``window`` is the number of positions each query attends to ending at itself
    (``window >= 1``). ``sink`` is the number of leading tokens every query
    attends to (``sink >= 0``).
    """

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
    """Build an SWA-with-sinks additive attention mask.

    Parameters
    ----------
    seq_q, seq_k
        Query and key sequence lengths. ``seq_q <= seq_k``. When ``seq_q == seq_k``
        this is prefill (each query attends to itself and earlier keys); when
        ``seq_q < seq_k`` this is decoding (new queries attend to ``seq_k`` past keys).
    window, sink
        SWA hyper-parameters as defined in :class:`Config`.
    device, dtype
        Output device and dtype.

    Returns
    -------
    torch.Tensor
        A ``(1, 1, seq_q, seq_k)`` additive mask in which ``0`` means attend and
        ``-inf`` means mask out.

    Notes
    -----
    A query at absolute position ``q`` attends to key positions
    ``[0, sink)`` union ``[q - window + 1, q]``. This is position-accurate and is
    used by :func:`swa_mask_function` for the model-level patch; this standalone
    helper exists for testing and for callers that want a raw mask tensor.
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


def swa_mask_function(window: int, sink: int) -> Callable[[int, int, int, int], bool]:
    """Return a scalar mask predicate for SWA-with-sinks.

    The returned callable has the ``(batch_idx, head_idx, q_idx, kv_idx) -> bool``
    signature expected by ``transformers.masking_utils`` mask factories and is
    built from that library's own vmap-safe combinators (:func:`transformers.
    masking_utils.or_masks`, :func:`transformers.masking_utils.and_masks`,
    :func:`transformers.masking_utils.sliding_window_overlay`,
    :func:`transformers.masking_utils.causal_mask_function`).

    ``q_idx`` and ``kv_idx`` are **absolute** positions (query positions come from
    ``cache_position``, key positions from ``kv_offset + arange(kv_length)``), so
    sinks and the sliding window are computed position-accurately and remain
    correct under padding and cache shifts.

    A query at absolute position ``q`` attends to keys in ``[0, sink)`` union
    ``[q - window + 1, q]`` (``window`` positions ending at ``q``, inclusive).
    """
    from transformers import masking_utils

    def sink_allows(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return kv_idx < sink

    windowed_causal = masking_utils.and_masks(
        masking_utils.causal_mask_function,
        masking_utils.sliding_window_overlay(window),
    )
    return masking_utils.or_masks(windowed_causal, sink_allows)


def swa_create_causal_mask(upstream: Callable):
    """Build the per-model replacement for ``transformers.masking_utils.create_causal_mask``.

    ``upstream`` is the original (unpatched) mask function this call replaces;
    it is captured in the returned closure so no global or aliased state is
    involved. The returned builder decides per call, from the ``config`` it is
    passed:

    * eager attention **and** ``config._beacon`` present (a patched model) →
      build the SWA-with-sinks 4-D additive mask via
      :func:`swa_mask_function`, mirroring the upstream eager tail and combining
      padding / packed-sequence masks exactly as upstream does;
    * anything else (unpatched model sharing the module binding, or a
      non-eager implementation) → delegate to ``upstream`` unchanged.

    Because ``window``/``sink`` are read from each model's own ``config``, every
    model uses its own :class:`Config` even though the architecture module may be
    shared across instances.
    """
    from transformers import masking_utils

    def create_causal_mask(
        config,
        input_embeds,
        attention_mask,
        cache_position,
        past_key_values,
        position_ids=None,
        or_mask_function=None,
        and_mask_function=None,
    ):
        beacon = getattr(config, "_beacon", None)
        if config._attn_implementation != "eager" or beacon is None:
            return upstream(
                config,
                input_embeds,
                attention_mask,
                cache_position,
                past_key_values,
                position_ids,
                or_mask_function,
                and_mask_function,
            )

        (
            early_exit,
            attention_mask,
            packed_sequence_mask,
            kv_length,
            kv_offset,
        ) = masking_utils._preprocess_mask_arguments(
            config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, layer_idx=0
        )
        if early_exit:
            return attention_mask

        batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
        mask_interface = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["eager"]
        mask_factory = swa_mask_function(beacon["window"], beacon["sink"])

        if or_mask_function is not None:
            mask_factory = masking_utils.or_masks(mask_factory, or_mask_function)
        if and_mask_function is not None:
            mask_factory = masking_utils.and_masks(mask_factory, and_mask_function)
        if packed_sequence_mask is not None:
            mask_factory = masking_utils.and_masks(
                mask_factory, masking_utils.packed_sequence_mask_function(packed_sequence_mask)
            )

        return mask_interface(
            batch_size=batch_size,
            cache_position=cache_position,
            kv_length=kv_length,
            kv_offset=kv_offset,
            mask_function=mask_factory,
            attention_mask=attention_mask,
            allow_is_causal_skip=False,
            dtype=dtype,
            config=config,
        )

    functools.wraps(upstream)(create_causal_mask)
    return create_causal_mask


def architecture_module(model: nn.Module):
    """Return the module that defines ``model``'s class (e.g. ``modeling_llama``)."""
    module_name = type(model).__module__
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - defensive
        raise NotImplementedError(
            f"cannot locate the architecture module {module_name!r} for {type(model).__name__}"
        ) from exc


def transformers_version() -> tuple[int, int]:
    """Return ``transformers.__version__`` as a ``(major, minor)`` tuple."""
    from transformers import __version__

    parts = []
    for tok in __version__.split(".")[:2]:
        digits = "".join(ch for ch in tok if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    if len(parts) < 2:
        raise RuntimeError(f"cannot parse transformers version {__version__!r}")
    return parts[0], parts[1]


def check_transformers_version() -> None:
    """Raise ``RuntimeError`` if the installed transformers is unsupported."""
    actual = transformers_version()
    if actual < MIN_TRANSFORMERS:
        raise RuntimeError(
            f"transformers {'.'.join(map(str, actual))} is unsupported; beacon requires "
            f">= {'.'.join(map(str, MIN_TRANSFORMERS))} (the masking_utils.create_causal_mask "
            "attention-pipeline era)"
        )


def patch(model: nn.Module, cfg: Config) -> nn.Module:
    """Make ``model`` attend with SWA-with-sinks instead of full causal attention.

    - Validates that the model and transformers version are supported; raises an
      explicit error otherwise (no silent full-causal fallback).
    - Forces the ``eager`` attention implementation so the 4-D additive mask is
      honoured.
    - Replaces ``create_causal_mask`` on the model's own architecture module with
      a per-model builder (the previous binding is captured as its upstream).
    - Records ``model.config._beacon`` so every run carries machine-readable proof
      the patch is active and which ``window``/``sink`` it uses.

    Returns ``model`` unchanged (no wrapper). Each call is independent: the
    builder reads each model's ``config._beacon`` per call, so patching two models
    (even of the same architecture) with different ``Config`` values does not
    cross-contaminate them.
    """
    from transformers import PreTrainedModel

    check_transformers_version()
    if not isinstance(model, PreTrainedModel):
        raise TypeError(
            f"patch expects a transformers PreTrainedModel, got {type(model).__name__}"
        )

    module = architecture_module(model)
    upstream = getattr(module, "create_causal_mask", None)
    if upstream is None:
        raise NotImplementedError(
            f"architecture {type(model).__name__} (module {module.__name__}) does not "
            "route attention through transformers.masking_utils.create_causal_mask; "
            "the SWA-with-sinks patch is unsupported for it"
        )

    model.config._attn_implementation = "eager"
    module.create_causal_mask = swa_create_causal_mask(upstream)
    model.config._beacon = {
        "window": cfg.window,
        "sink": cfg.sink,
        "applied": True,
        "transformers_version": transformers_version(),
    }
    return model