# Methods

This document explains the one algorithmic idea — **Sliding Window Attention
with sinks** — and how it is wired into `transformers`, so a reviewer can
verify the reproduction is faithful to the paper and not a silent fallback to
full causal attention.

## The mask

`beacon.patch.mask(seq_q, seq_k, window, sink)` returns the additive attention
mask. A query at absolute position `q` attends to keys:

```
[0, sink)  ∪  [q - window + 1, q]        (both ranges position-clamped)
```

- `window` counts the current position (inclusive): `q` always attends to
  itself and the `window - 1` positions before it.
- `sink` "beacon" positions at the very start are global: every query attends
  to them, even queries that would otherwise look *ahead* into the first
  `sink` tokens (early queries may attend later sink positions by design).

Because the mask is built from vmap-safe combinators from
`transformers.masking_utils` (`or_masks`, `and_masks`, `sliding_window_overlay`,
`causal_mask_function`), it composes correctly with the batched eager backend.

### Degenerate cases

- `window >= seq_k` → lower-triangular causal plus the global sink columns
  (future tokens are still blocked, except sinks are reachable).
- `sink == seq_k` → every row is fully attended (sinks cover the whole prefix).
- `sink == 0, window == 1` → only the diagonal is attended.

## How the patch reaches the real attention path

Modern `transformers` (>= 4.55) builds attention masks in
`transformers.masking_utils.create_causal_mask`, which each `modeling_<arch>`.
module imports by reference and calls from `forward`. The `PreTrainedModel._update_causal_mask`
hook that older reproductions patched is **no longer called** during forward, so
patching it silently keeps full causal attention.

`beacon.patch.patch(model, cfg)` instead:

1. Verifies `transformers.__version__ >= 4.55` (`MIN_TRANSFORMERS`), else raises.
2. Forces `_attn_implementation = "eager"` (the SWA mask function is vmap-built,
   so it cannot mix with sdpa/flash fused-kernels).
3. Resolves the model's own architecture module via
   `beacon.patch.architecture_module(model)` — i.e. `importlib` of
   `type(model).__module__` — and checks it actually defines
   `create_causal_mask`; otherwise raises `NotImplementedError` (e.g. encoder
   models).
4. Installs a per-model SWA builder: `module.create_causal_mask` is replaced by
   `swa_create_causal_mask(upstream)`, a closure bound to *that* model's
   `config._beacon = {"window", "sink", "applied": True, ...}`.

`config._beacon` is read per model on every forward, so two patched models of
the same architecture each get their own window/sink — there is no shared
global "last patch" state, and patching one model never leaks into another.

## Running against full causal attention

To compare against the paper's baseline you do not patch at all:

```python
from beacon.speed import fa
model = fa(model, None)  # pass-through, full causal attention
```

Or use the `fa` speed method, whose KV cost grows linearly with context, versus
`swa`, whose steady-state KV is bounded by `window + sink` (`speed.kv_swa`).

## Why only eager?

`torch.nn.functional.attention` / SDPA and flash-attention kernels take closed-form
`attn_mask` tensors only when they match a small set of fused patterns. An
arbitrary SWA-with-sinks mask from `swa_mask_function` does not, so mixing it
into sdpa/flash would degrade to a dense unfold or an error. The eager backend
materialises the additive mask and applies it per head, which is exactly the
generality SWA-with-sinks needs. The throughput numbers in Figure 2 therefore
compare eager-SWA against eager-full-causal on equal footing.

## Version bounds

`transformers>=4.55` is required at install time (`pyproject.toml`) and enforced
again at runtime. Below 4.55, `masking_utils.create_causal_mask` does not exist
and the reproduction degrades silently — which is precisely what this project
refuses to do.