"""Centralized, deterministic seeding.

Benchmarks must be reproducible: the same ``seed`` should give the same input
tokens, model init and decode trajectory. A single entry point keeps every RNG
source (Python ``random``, NumPy, torch CPU and per-device torch CUDA) in
lockstep instead of scattering ``torch.manual_seed`` calls per module.
"""

from __future__ import annotations

import random

import torch


def seed_all(seed: int) -> None:
    """Seed every RNG source (Python, NumPy, torch CPU and CUDA) for ``seed``."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dep but stay safe
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seeded_rng(seed: int) -> random.Random:
    """A fresh ``random.Random`` split from ``seed`` for independent generation.

    Use this inside per-cell sample generation so each cell gets a stable RNG
    without mutating the global ``random`` state that ``seed_all`` owns.
    """
    return random.Random(seed)


__all__ = ["seed_all", "seeded_rng"]