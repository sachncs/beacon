"""Tests for beacon.sample — Sample, filler, insert, contains."""

import random
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from beacon.sample import Sample, contains, filler, insert  # noqa: E402


def test_filler_word_count_within_tolerance():
    rng = random.Random(0)
    text = filler(rng, 200)
    n = len(text.split())
    assert abs(n - 200) <= 25, f"filler produced {n} words for n_words=200"


def test_insert_endpoints():
    rng = random.Random(0)
    hay = filler(rng, 80)
    needle = "THE_NEEDLE"
    # frac=0 -> needle at start (after first token)
    head = insert(hay, needle, 0.0)
    assert head.split().index("THE_NEEDLE") <= 1
    # frac=1.0 -> needle near end
    tail = insert(hay, needle, 1.0)
    assert tail.split().index("THE_NEEDLE") >= len(tail.split()) - 5


def test_insert_fraction_middle():
    rng = random.Random(0)
    hay = filler(rng, 100)
    mid = insert(hay, "MARKER", 0.5)
    pos = mid.split().index("MARKER")
    assert 30 <= pos <= 70, f"MARKER at pos {pos}, expected ~50"


def test_contains_case_insensitive():
    assert contains("cat", "the CAT sat")
    assert contains("CAT", "the cat sat")
    assert contains("dog", "DOG days")
    assert contains("cat", "the dog sat") is False


def test_contains_whitespace_insensitive():
    assert contains("hello world", "hello,\n  world!")
    assert contains("hello, world", "hello world")


def test_contains_empty_answer_is_false():
    assert contains("", "anything") is False
    assert contains("   ", "anything") is False


def test_sample_is_frozen():
    s = Sample(prompt="p", answer="a", meta={"k": 1})
    assert s.prompt == "p"
    assert s.answer == "a"
    assert s.meta == {"k": 1}
    try:
        s.prompt = "other"
        raise AssertionError("Sample is mutable")
    except FrozenInstanceError:
        pass


if __name__ == "__main__":
    test_filler_word_count_within_tolerance()
    test_insert_endpoints()
    test_insert_fraction_middle()
    test_contains_case_insensitive()
    test_contains_whitespace_insensitive()
    test_contains_empty_answer_is_false()
    test_sample_is_frozen()
    print("All sample tests passed.")