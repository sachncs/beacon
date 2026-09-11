"""Tests for beacon.seed — seed_all / seeded_rng determinism."""

import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from beacon.seed import seed_all, seeded_rng


def test_seed_all_reproduces_torch_stream():
    def stream():
        seed_all(123)
        return torch.randn(5), [random.random() for _ in range(3)]

    a_bits, a_py = stream()
    b_bits, b_py = stream()
    assert torch.equal(a_bits, b_bits)
    assert a_py == b_py


def test_seed_all_prime_side_effects_cpu():
    # Regression: don't depend on leftover RNG from a prior test.
    seed_all(1)
    first = torch.randn(3)
    seed_all(99)
    seed_all(1)
    assert torch.equal(first, torch.randn(3))


def test_seeded_rng_is_deterministic_and_isolated():
    r1, r2 = seeded_rng(7), seeded_rng(7)
    assert r1.random() == r2.random()
    assert r1.choice("abcd") == r2.choice("abcd")
    r_other = seeded_rng(8)
    assert r_other.random() != r1.random()


def test_seeded_rng_does_not_touch_global_state():
    seed_all(0)
    before = random.random()
    seeded_rng(123)
    seed_all(0)
    assert random.random() == before


if __name__ == "__main__":
    test_seed_all_reproduces_torch_stream()
    test_seed_all_prime_side_effects_cpu()
    test_seeded_rng_is_deterministic_and_isolated()
    test_seeded_rng_does_not_touch_global_state()
    print("All seed tests passed.")
