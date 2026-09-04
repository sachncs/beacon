"""Tests for beacon.story — TASK registry, samples(), task generators."""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from beacon.story import TASK, qa1, qa2, qa3, qa4, qa5, samples  # noqa: E402


def test_task_registry_has_five():
    assert set(TASK.keys()) == {1, 2, 3, 4, 5}


def test_qa1_single_fact():
    rng = random.Random(0)
    for _ in range(20):
        story, question, answer = qa1(rng)
        assert answer in story
        assert question.startswith("Where is")


def test_qa2_travel():
    rng = random.Random(0)
    story, question, answer = qa2(rng)
    assert answer in story
    assert "travelled" in story


def test_qa3_object_movement():
    rng = random.Random(0)
    story, question, answer = qa3(rng)
    assert answer in story
    assert "picked up" in story and "dropped" in story


def test_qa4_yes_no():
    rng = random.Random(0)
    story, question, answer = qa4(rng)
    assert answer == "no"
    assert question.startswith("Is ")


def test_qa5_counting():
    rng = random.Random(0)
    story, question, answer = qa5(rng)
    items = answer.split()
    assert len(items) >= 2
    assert all(item in story for item in items)


def test_samples_grid_shape():
    cell = samples(task=[1, 3, 5], ctx=[100, 200], frac=[0.5], seed=0, n=2)
    assert len(cell) == 3 * 2 * 1 * 2
    for s in cell:
        assert s.meta["task"] in (1, 3, 5)
        assert s.meta["ctx"] in (100, 200)


def test_samples_unknown_task_raises():
    with pytest.raises(ValueError):
        samples(task=[99], ctx=[10])


def test_samples_are_deterministic():
    a = samples(task=[1], ctx=[80], seed=7, n=1)
    b = samples(task=[1], ctx=[80], seed=7, n=1)
    assert a[0].prompt == b[0].prompt


def test_samples_ctx_beyond_8192_is_not_capped():
    """Regression: prompts must not be silently truncated to 8192 tokens."""
    # Build the prompt at a real >8192-word context and check it keeps the story.
    from beacon.sample import contains

    cell = samples(task=[1], ctx=[20000], seed=3, n=1)
    s = cell[0]
    assert len(s.prompt.split()) > 8192
    assert contains(s.answer, s.prompt)


if __name__ == "__main__":
    test_task_registry_has_five()
    test_qa1_single_fact()
    test_qa2_travel()
    test_qa3_object_movement()
    test_qa4_yes_no()
    test_qa5_counting()
    test_samples_grid_shape()
    test_samples_unknown_task_raises()
    test_samples_are_deterministic()
    print("All story tests passed.")