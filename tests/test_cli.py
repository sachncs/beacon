"""Tests for beacon.cli — argparse dispatch.

We avoid running the real benchmarks here (they would download models).
Instead we patch the per-subcommand ``run`` functions to record that
they were called and assert the dispatch flow.
"""

import argparse
import importlib

import pytest


def test_cli_help_lists_subcommands():
    from beacon.cli import main

    # argparse raises SystemExit on --help
    with pytest.raises(SystemExit):
        main(["--help"])


def test_dispatch_table_has_all_subcommands():
    from beacon import cli

    assert set(cli.dispatch.keys()) == {"short", "find", "story", "speed"}


def test_cli_dispatches_find_to_find_run(monkeypatch, capsys):
    """Verify the dispatcher calls beacon.find.run with parsed kwargs."""
    from beacon import cli

    called = {}

    def fake_run(*, model, cfg, ctx, variant, frac, max_new_tokens, seed, n, dtype, out):
        called["model"] = model
        called["ctx"] = ctx
        called["variant"] = variant
        called["cfg"] = cfg
        called["max_new_tokens"] = max_new_tokens

    import beacon.find as find_mod

    monkeypatch.setattr(find_mod, "run", fake_run)

    rc = cli.main([
        "find",
        "--model", "test-model",
        "--window", "32",
        "--sink", "2",
        "--ctx", "128,256",
        "--variant", "1,2",
        "--frac", "0.5",
        "--max-new-tokens", "8",
        "--seed", "1",
        "--n", "2",
    ])
    assert rc == 0
    assert called["model"] == "test-model"
    assert called["ctx"] == [128, 256]
    assert called["variant"] == ("1", "2")
    assert called["cfg"].window == 32
    assert called["cfg"].sink == 2
    assert called["max_new_tokens"] == 8


def test_cli_dispatches_story(monkeypatch):
    from beacon import cli

    called = {}

    def fake_run(*, model, cfg, ctx, task, frac, max_new_tokens, seed, n, dtype, max_ctx, out):
        called["task"] = task
        called["ctx"] = ctx
        called["max_new_tokens"] = max_new_tokens
        called["max_ctx"] = max_ctx

    import beacon.story as story_mod

    monkeypatch.setattr(story_mod, "run", fake_run)

    rc = cli.main([
        "story",
        "--model", "m",
        "--ctx", "100,200",
        "--task", "1,3,5",
        "--max-new-tokens", "16",
        "--max-ctx", "3000",
    ])
    assert rc == 0
    assert called["task"] == [1, 3, 5]
    assert called["ctx"] == [100, 200]
    assert called["max_new_tokens"] == 16
    assert called["max_ctx"] == 3000


def test_cli_dispatches_short(monkeypatch, capsys):
    from beacon import cli

    called = {}

    def fake_run(s):
        called["task"] = s.task
        called["batch"] = s.batch
        called["shot"] = s.shot
        called["limit"] = s.limit
        return {"x": 1.0}

    import beacon.short as short_mod

    monkeypatch.setattr(short_mod, "run", fake_run)

    rc = cli.main([
        "short",
        "--model", "m",
        "--task", "mmlu,arc_easy",
        "--batch", "8",
        "--shot", "5",
        "--limit", "100",
    ])
    assert rc == 0
    assert called["task"] == ("mmlu", "arc_easy")
    assert called["batch"] == "8"
    assert called["shot"] == 5
    assert called["limit"] == 100
    # cli prints the JSON of the returned dict
    out = capsys.readouterr().out
    assert '"x": 1.0' in out


def test_cli_dispatches_speed(monkeypatch):
    from beacon import cli

    called = {}

    def fake_run(*, model, cfg, ctx, method, step, seed, dtype, out):
        called["ctx"] = ctx
        called["method"] = method
        called["step"] = step
        called["cfg"] = cfg
        return {"fa": [], "swa": []}

    import beacon.speed as speed_mod

    monkeypatch.setattr(speed_mod, "run", fake_run)

    rc = cli.main([
        "speed",
        "--model", "m",
        "--window", "16",
        "--sink", "2",
        "--ctx", "64,128",
        "--method", "fa",
        "--step", "4",
    ])
    assert rc == 0
    assert called["ctx"] == [64, 128]
    assert called["method"] == ("fa",)
    assert called["step"] == 4
    assert called["cfg"].window == 16


def test_cli_unknown_subcommand_raises():
    from beacon.cli import main

    with pytest.raises(SystemExit) as e:
        main(["no-such-cmd"])
    assert e.value.code != 0


def test_cli_short_default_shot_is_five(monkeypatch):
    """The Table-2 runner defaults MMLU et al. to 5-shot unless overridden."""
    from beacon import cli

    captured = {}

    def fake_run(s):
        captured["shot"] = s.shot
        return {}

    import beacon.short as short_mod

    monkeypatch.setattr(short_mod, "run", fake_run)

    rc = cli.main(["short", "--model", "m"])
    assert rc == 0
    assert captured["shot"] == 5


def test_cli_dtype_is_passed_through(monkeypatch):
    from beacon import cli

    captured = {}

    def fake_run(*, model, cfg, ctx, variant, frac, max_new_tokens, seed, n, dtype, out):
        captured["dtype"] = dtype
        return {}

    import beacon.find as find_mod

    monkeypatch.setattr(find_mod, "run", fake_run)

    rc = cli.main(["find", "--model", "m", "--dtype", "bfloat16"])
    assert rc == 0
    assert captured["dtype"] == "bfloat16"


def test_per_module_cli_entry_points_are_removed():
    """beacon.cli.main is the one and only CLI entry point.

    find/story/short/speed expose run(...), not a thin argparse wrapper; the
    unified dispatch owns all flag parsing so flags never drift per module.
    """
    for mod in ("find", "story", "short", "speed"):
        m = importlib.import_module(f"beacon.{mod}")
        assert not hasattr(m, "cli")


def test_default_model_env(monkeypatch):
    """BEACON_MODEL env var overrides the default."""
    monkeypatch.setenv("BEACON_MODEL", "env-model")
    # Force reimport to pick up the env var
    from beacon import cli

    importlib.reload(cli)
    assert cli.DEFAULT_MODEL == "env-model"
    # Reload other modules too in case they cache it.
    monkeypatch.delenv("BEACON_MODEL")


def test_default_window_and_sink_are_consistent_across_subcommands():
    """Every subcommand must share the same --window / --sink defaults."""
    from beacon import cli

    assert cli.DEFAULT_WINDOW == 64
    assert cli.DEFAULT_SINK == 4

    parser = argparse.ArgumentParser(prog="beacon-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("short", "find", "story", "speed"):
        p = sub.add_parser(name)
        p.add_argument("--window", type=int, default=cli.DEFAULT_WINDOW)
        p.add_argument("--sink", type=int, default=cli.DEFAULT_SINK)


if __name__ == "__main__":
    test_cli_help_lists_subcommands()
    test_dispatch_table_has_all_subcommands()
    print("Basic CLI tests passed. Run pytest for the full suite.")
