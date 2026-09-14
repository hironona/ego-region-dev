# ego_region

Where does a chat model represent *who is speaking*, and is that representation load-bearing?
Four experiments on `Qwen/Qwen3-0.6B`: probe for the user/assistant direction, localise it by
ablation, then test whether a trait steering vector interacts with the boundary it defines.

## Setup

```bash
uv sync
```

Device is auto-detected; `--device cpu|cuda|mps` overrides. Use `cpu` or `cuda` for real
runs — TransformerLens reports MPS can be silently wrong.

## Experiments

| folder | question | headline |
|---|---|---|
| `speaker_probe` | is user vs assistant linearly decodable from the residual stream? | ~0.98 balanced accuracy from layer 1; the direction is near-orthogonal to the top PCs |
| `mean_ablation_probe` | which blocks write it? | attention blocks 0–4 |
| `head_ablation_sweep` | which heads? | per-head sweep over that window |
| `steering_boundary` | does trait steering bend near the probe's boundary `w·h + b = 0`? | open — see `docs/HANDOFF.md` |

Each runs the same way: a capture step that loads the model, then analysis that only reads
the saved `.npz`, so plots are free to regenerate.

```bash
uv run python -m experiments.speaker_probe.run_capture
uv run python -m experiments.speaker_probe.analyze
```

`steering_boundary` has an extra sweep step between the two, and `analyze` is run twice —
once to build the vector and size the grid, once after the sweep:

```bash
uv run python -m experiments.steering_boundary.run_capture --device cuda
uv run python -m experiments.steering_boundary.analyze          # writes vectors.npz
uv run python -m experiments.steering_boundary.run_steer --device cuda
uv run python -m experiments.steering_boundary.analyze          # plots + results.json
```

Outputs land in `experiments/<name>/outputs/` (gitignored): captures plus `plots/*.png`.
Captures are large — 160 MB to 1.3 GB per experiment.

## What is captured

Per forward pass: `hidden_states` `(L+1, T, D)` (index 0 is the embedding output, then each
layer's `resid_post`), `attentions` `(L, H, T, T)`, `values` `(L, T, D_kv)`, and `logits`
`(T, V)` — selected via `what=`, with optional write hooks for ablation and steering. A
sidecar `.meta.json` records the templated text, token strings and label positions.

## Checks

```bash
uv run python test_pipeline.py
```

11 asserts covering npz round-trip, capture shapes, the two Qwen3 `<think>` template
quirks, the lexical-confound control, and that each intervention touches exactly the
tokens it claims to.

## More

`docs/HANDOFF.md` — findings, traps, and open items. [CLAUDE.md](CLAUDE.md) — architecture
rules; read before adding an experiment.
