# Changelog

All notable changes to **beacon** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- `speed.kv()` and `speed.kv_swa()` use `num_kv_heads * head_dim` instead of
  `hidden_size`; the previous formula overcounted KV-cache memory on GQA
  models (Llama, MiniCPM5, Qwen, Gemma) by a factor of
  `num_attention_heads / num_key_value_heads`.
- `short.run()` picks `acc_norm` for HellaSwag / ARC / PIQA — the
  length-normalised figure is the paper-comparable metric, not raw `acc`.
- `story.qa4` now produces a balanced mix of `yes`/`no` answers; the
  previous generator always answered `no`, letting a trivial baseline hit 100%.
- `sample.contains()` normalises underscores so `hello_world` matches
  `hello world` (Python `\w` includes `_`, which leaked into both sides).
- `transformers_version()` parses PEP 440 suffixes correctly; previously it
  silently dropped trailing digits after a `rc`/`post`/`dev` marker.
- `patch()` records `previous_attn_implementation` in the stamp so callers
  can see what the eager override replaced.

### Added
- `patch.unpatch(model)` restores the upstream `create_causal_mask` and
  clears `config._beacon`.
- `speed.run()` loads the model once and reuses it across methods instead
  of reloading per method.
- Tool configuration: `tool.ruff`, `tool.mypy`, `tool.pytest.ini_options` in
  `pyproject.toml`; `ruff` and `mypy` added to the `dev` extras.

### Removed
- Unused direct dependencies: `accelerate`, `datasets`.
- Per-test `sys.path.insert` (conftest already does it).

## [0.2.0] - 2026-09-03

Initial public release of the SWA-with-sinks reproduction:

- `Config`, `mask`, `patch`, `load` public API.
- `find` (NIAH), `story` (BABILong), `short` (MMLU/ARC/HellaSwag/PIQA/WinoGrande),
  `speed` (Figure 2) runners.
- `beacon-eval` unified CLI (`short` / `find` / `story` / `speed`).
- Tests, CI workflow, OSS community files.

[Unreleased]: https://github.com/sachncs/beacon/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/sachncs/beacon/releases/tag/v0.2.0