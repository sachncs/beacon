# beacon

A reproduction of **"Sliding-window beats linear attention"** (Jolicoeur-Martineau et al., 2026).

The paper's claim: a plain pretrained Transformer with its attention mask replaced by
**Sliding Window Attention + attention sinks** matches or beats post-trained linear-attention
models on knowledge/reasoning benchmarks, and crushes them on long-context tasks — with
zero post-training and lower memory.

**beacon** = the SWA-with-sinks mechanism. Sink tokens act as beacons anchoring attention
while the rest of the context scrolls past.

## Install

```bash
pip install -e ".[eval]"
```

## Layout

```
beacon/
  __init__.py   # public API: Config, patch, mask, load
  patch.py      # Config, mask(), patch()  — single file, the algorithm
  load.py       # load(model, cfg) → (model, tokenizer)
  sample.py     # Sample, filler, insert, contains
  seed.py       # seed_all(), seeded_rng()  — central determinism
  short.py      # Short dataclass + run() via lm-eval (Table 2)
  find.py       # VARIANT registry + run() (S-NIAH, Table 3)
  story.py      # TASK registry + run()  (BABILong, Table 4)
  speed.py      # Method registry + run() + kv()/kv_swa() (Figure 2)
  cli.py        # beacon-eval: the single CLI entry point (short/find/story/speed)
tests/
  test_patch.py test_sample.py test_find.py
  test_story.py test_speed.py test_cli.py test_load.py test_seed.py
```

## Quick start

Default model: **`openbmb/MiniCPM5-1B`** — a standard `LlamaForCausalLM` (GQA 16/2, 24 layers,
131K context). Override via the `BEACON_MODEL` env var or `--model` flag.

```bash
# short-context eval (Table 2)
python -m beacon.cli short --model openbmb/MiniCPM5-1B \
    --window 64 --sink 4 --task mmlu,arc_easy,hellaswag,piqa,winogrande

# Single Needle-in-a-Haystack (Table 3)
python -m beacon.cli find --model openbmb/MiniCPM5-1B \
    --window 256 --sink 4 --variant 1,2,3

# BABILong (Table 4)
python -m beacon.cli story --model openbmb/MiniCPM5-1B \
    --window 256 --sink 4 --task 1,2,3,4,5

# Speed / KV-cache memory (Figure 2)
python -m beacon.cli speed --model openbmb/MiniCPM5-1B \
    --window 64 --sink 4 --method fa,swa
```

### What the output looks like

Each subcommand prints one line per cell to stdout (the JSON dump in
`--out <file>` carries the same numbers). The shapes you should expect:

```
# short — Table 2 numbers (per task, 5-shot, acc_norm for HellaSwag/ARC/PIQA)
{"mmlu": 0.404, "arc_challenge": 0.341, "arc_easy": 0.628,
 "hellaswag": 0.581, "piqa": 0.745, "winogrande": 0.602}

# find — NIAH-{variant} ctx={ctx}: acc={acc}
NIAH-1 ctx=1024: acc=1.000
NIAH-2 ctx=4096: acc=0.667
NIAH-3 ctx=2048: acc=0.333

# story — average + per-task accuracy at each context length
BABILong ctx=2000 avg=0.733 per-task={1: 0.9, 2: 0.8, 3: 0.7, 4: 0.7, 5: 0.6}

# speed — one row per (method, ctx): KV(prefill) and KV(steady) in MiB
=== fa ===
fa  ctx=   4096  tps=  42.3  latency=23.65ms  KV(prefill)=128.00 MiB  KV(steady)=128.00 MiB
=== swa ===
swa ctx=  4096  tps=  38.1  latency=26.24ms  KV(prefill)=128.00 MiB  KV(steady)=  0.36 MiB
```

Numbers are illustrative; they show the *shape* of the output, not a
benchmark. Run the commands to see the actual numbers on your hardware.

## Using the patch directly

```python
from beacon import Config, patch, load

model, tok = load("openbmb/MiniCPM5-1B", Config(window=64, sink=4))
# model is already patched and in eval mode.
```

### MiniCPM5 chat template

```python
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "..."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
```

## Extension points

Every polymorphism registry is a plain Python dict — add a new variant / task /
method by adding one entry:

```python
# Add a new NIAH variant
from beacon.find import VARIANT
def v4(rng): ...  # returns (needle, question, answer)
VARIANT["4"] = v4

# Add a new bAbI task
from beacon.story import TASK
def qa6(rng): ...
TASK[6] = qa6

# Add a new attention method (e.g. custom kernel)
from beacon.speed import METHOD, Method
def my_linear(model, cfg): ...
METHOD["linear"] = Method("linear", my_linear)
```

## Design principles

- **Public API is tiny.** `Config`, `mask`, `patch`, `load`. Plus the per-module
  `run` functions and strategy registries (`VARIANT`, `TASK`, `METHOD`).
- **Single-word naming everywhere.** No `_helper`, no `window_size`, no
  `_make_something`. Each name says what it is.
- **Polymorphism via registries, not class hierarchies.** Real behavioural
  variation lives in `dict`s of callables.
- **Frozen dataclasses for configurations.** `Short`, `Config` are immutable
  by default.
- **Determinism.** Every RNG-taking function takes an explicit `seed`, and all
  seeding funnels through `beacon.seed.seed_all()` / `seeded_rng()`. The speed
  bench prefills a fixed context and times steady-state decode with a
  per-device `torch.Generator`.

## CLI flags (single-word)

```
short : --model --window --sink --task --batch --shot --limit --dtype --out
find  : --model --window --sink --ctx --variant --frac --max-new-tokens --seed --n --dtype --out
story : --model --window --sink --ctx --task --frac --max-new-tokens --max-ctx --seed --n --dtype --out
speed : --model --window --sink --ctx --method --step --seed --dtype --out
```

`story --max-ctx` bounds the input token length (prompts are **not** truncated
by default); `--shot` defaults to 5 for Table 2 (MMLU is reported 5-shot).

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

## Paper-vs-ours differences

- **Default model**: `openbmb/MiniCPM5-1B` (matches the paper's LlamaForCausalLM
  architectural profile, including GQA).
- **Hardware**: unverified. The speed benchmark (`beacon.speed.run`) reports
  whatever the local device produces; throughput numbers in this README and
  in `docs/METHODS.md` are illustrative, not authoritative.
- **BABILong**: reimplemented from scratch (no `babilong` PyPI dep) — generators
  reproduce the same story structure; filler text is synthetic instead of PG essays.

## Citation

```
@inproceedings{jolicoeurmartineau2026sliding,
  title  = {Sliding-window beats linear attention},
  author = {Jolicoeur-Martineau, Alexia and Sukthanker, Rhea Sanjay and Cameron, Pashmina and Gervais, Emy},
  year   = {2026}
}
```