"""Tests for beacon.load — dtype coercion.

The heavy ``load()`` itself downloads a model, so these cover the pure,
download-free pieces: ``_coerce_dtype`` normalization and validation.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from beacon.load import _coerce_dtype


def test_dtype_passthrough_returns_same():
    assert _coerce_dtype(torch.float16) is torch.float16
    assert _coerce_dtype(torch.bfloat16) is torch.bfloat16


def test_dtype_string_names():
    assert _coerce_dtype("float16") is torch.float16
    assert _coerce_dtype("bfloat16") is torch.bfloat16
    assert _coerce_dtype("float32") is torch.float32
    assert _coerce_dtype("float64") is torch.float64


def test_dtype_string_aliases():
    assert _coerce_dtype("half") is torch.float16
    assert _coerce_dtype("float") is torch.float32
    assert _coerce_dtype("double") is torch.float64


def test_dtype_unknown_string_raises():
    with pytest.raises(ValueError):
        _coerce_dtype("float17")


def test_dtype_non_string_type_raises():
    with pytest.raises(TypeError):
        _coerce_dtype(3.5)


if __name__ == "__main__":
    test_dtype_passthrough_returns_same()
    test_dtype_string_names()
    test_dtype_string_aliases()
    test_dtype_unknown_string_raises()
    test_dtype_non_string_type_raises()
    print("All load tests passed.")