# beacon

A reproduction of **"Sliding-window beats linear attention"** (Jolicoeur-Martineau et al., 2026).

The paper's claim: a plain pretrained Transformer with its attention mask replaced by
**Sliding Window Attention + attention sinks** matches or beats post-trained linear-attention
models on knowledge/reasoning benchmarks, and crushes them on long-context tasks — with
zero post-training and lower memory.

This repo gives you a model-agnostic drop-in patch plus the full evaluation suite used in
the paper (Table 1–4, Figure 2).

## What's here

| File | What it does |
|------|--------------|
| `beacon/patch.py`     | Core SWA-with-sinks mask + HF model patch |
| `beacon/short.py`     | MMLU, ARC-C, ARC-E, HellaSwag, PIQA, WinoGrande (Table 2) |
| `beacon/find.py`      | Single Needle-in-a-Haystack (Tables 3) |
| `beacon/story.py`     | BABILong (Table 4) |
| `beacon/speed.py`     | Throughput / KV-cache size vs context (Figure 2) |
| `beacon/cli.py`       | Unified CLI (`beacon-eval`) |

## Quick start

Default model is **openbmb/MiniCPM5-1B** (1B, standard `LlamaForCausalLM`,
GQA 16/2, 131K context — passes through the patch unchanged).

```bash
pip install -e ".[eval]"

# short-context eval (Table 2)
python -m beacon.cli short --model openbmb/MiniCPM5-1B \
    --window 64 --sinks 4 --tasks mmlu,arc_easy,hellaswag,piqa,winogrande

# long-context eval (Tables 3-4)
python -m beacon.cli find --model openbmb/MiniCPM5-1B \
    --window 256 --sinks 4

python -m beacon.cli story --model openbmb/MiniCPM5-1B \
    --window 256 --sinks 4

# speed/memory benchmark (Figure 2)
python -m beacon.cli speed --model openbmb/MiniCPM5-1B \
    --window 64 --sinks 4
```

## Using the patch directly

```python
from beacon import patch_swa
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("openbmb/MiniCPM5-1B")
model = patch_swa(model, window_size=64, num_sinks=4)
# model now uses SWA(64, 4) at every forward pass. No retraining.
```

### MiniCPM5 chat template

MiniCPM5 ships a chat template with `enable_thinking`. For instruction-style
evaluation, set it before tokenising the prompt:

```python
tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "..."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
```

## Paper-vs-ours differences

- **Default model**: `openbmb/MiniCPM5-1B` — a standard `LlamaForCausalLM`, so the
  SWA patch is drop-in. Pass `--model` to swap (e.g. `Qwen/Qwen2.5-1.5B-Instruct`,
  `meta-llama/Llama-3.1-8B`).
- **Hardware**: tested on Apple Silicon and CUDA. Speed benchmarks use the requested
  RTX PRO 6000 spec where available, fall back to local device otherwise.
- **BABILong**: reimplemented from scratch (no `babilong` PyPI dep) — generators
  reproduce the same story structure used in BABILong, but PG-essay filler is
  replaced with synthetic filler.

## Citation

```
@inproceedings{jolicoeurmartineau2026sliding,
  title  = {Sliding-window beats linear attention},
  author = {Jolicoeur-Martineau, Alexia and Sukthanker, Rhea Sanjay and Cameron, Pashmina and Gervais, Emy},
  year   = {2026}
}
```