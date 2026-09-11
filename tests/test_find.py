"""Tests for beacon.find — VARIANT registry, samples(), run() shape."""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from beacon.find import VARIANT, samples, v1, v2, v3


def test_variant_registry_has_three():
    assert set(VARIANT.keys()) == {"1", "2", "3"}


def test_v1_is_constant():
    rng = random.Random(0)
    assert v1(rng) == v1(rng)
    needle, question, answer = v1(None)
    assert answer in needle
    assert question in needle or "?" in question


def test_v2_contains_answer_in_needle():
    rng = random.Random(42)
    for _ in range(20):
        needle, question, answer = v2(rng)
        assert answer in needle
        assert answer in question or "?" in question


def test_v3_answer_is_a_key_in_needle():
    rng = random.Random(42)
    for _ in range(20):
        needle, question, answer = v3(rng)
        assert answer in needle
        # answer is one of the keys (8-char uppercase alphanumeric)
        assert any(part.startswith("If " + answer + " ") for part in needle.split(". ") if part)


def test_samples_grid_shape():
    cell = samples(ctx=[60, 120], frac=[0.0, 0.5, 1.0], variant="1", seed=0, n=2)
    assert len(cell) == 2 * 3 * 2  # ctx * frac * n
    for s in cell:
        assert s.meta["variant"] == "1"
        assert s.meta["ctx"] in (60, 120)
        assert s.answer == "sandwich"


def test_samples_unknown_variant_raises():
    with pytest.raises(ValueError):
        samples(ctx=[10], variant="9")


def test_samples_v2_deterministic_with_seed():
    a = samples(ctx=[80], variant="2", seed=123, n=1)
    b = samples(ctx=[80], variant="2", seed=123, n=1)
    assert a[0].prompt == b[0].prompt
    assert a[0].answer == b[0].answer


if __name__ == "__main__":
    test_variant_registry_has_three()
    test_v1_is_constant()
    test_v2_contains_answer_in_needle()
    test_v3_answer_is_a_key_in_needle()
    test_samples_grid_shape()
    test_samples_unknown_variant_raises()
    test_samples_v2_deterministic_with_seed()
    print("All find tests passed.")
