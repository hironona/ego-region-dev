# ego_region

Interpretability pipelines over activation geometry. First experiment:
**self_user_geometry** — do "self" (assistant) and "user" role tokens occupy
separable regions of a model's activation space?

## Setup

```bash
uv sync
```

Default model is `Qwen/Qwen3-0.6B` on `mps` (auto-detected; `--device cpu|cuda|mps` overrides).

## Run

Capture (loads the model once, writes raw activations to disk):

```bash
uv run python -m experiments.self_user_geometry.run_capture --prompt "Who are you, and who am I?"
```

Analyse (no model load — reruns freely on the saved capture):

```bash
uv run python -m experiments.self_user_geometry.analyze
```

Outputs land in `experiments/self_user_geometry/outputs/`:
`capture.npz` + `capture.meta.json`, and `plots/*.png`.

## What is captured

Per forward pass, for the chat-templated prompt:

| array | shape | contents |
|---|---|---|
| `hidden_states` | `(L+1, T, D)` | residual stream: index 0 is the embedding output, then `resid_post` of each layer (last row is pre-final-norm) |
| `attentions` | `(L, H, T, T)` | attention weights, from `hook_pattern` |
| `values` | `(L, T, D_kv)` | value vectors, from `hook_v` (GQA heads flattened) |

`capture.meta.json` records the templated text, token strings, and the **role-token
positions**, found by locating `<|im_start|>` in the tokenized prompt and reading the
token that actually follows it — verified against the tokenizer, not hardcoded. For
Qwen3 that resolves to `user` (id 872) and `assistant` (id 77091).

## Plots

- `layer_trajectories.png` — role tokens' path through a shared hidden-state PCA space, layer by layer (2D + 3D)
- `tokens_by_layer.png` — all tokens at first/middle/last layer, role tokens highlighted
- `value_directions.png` — value-vector directions at the role tokens in value-space PCA, plus their per-layer cosine similarity
- `role_attention.png` — attention received by each role token per layer

## Checks

```bash
uv run python test_pipeline.py
```

Covers role-token resolution against the real tokenizer, per-layer normalisation,
npz round-trip, and capture shapes (including that attention rows are causal and
sum to 1).

## Structure

Capture runs on TransformerLens `HookedTransformer` (loaded with
`from_pretrained_no_processing`, so activations match a plain HF forward pass).

`core/` is experiment-agnostic (model loading, hooks, capture, storage);
`experiments/<name>/` holds one experiment each. See [CLAUDE.md](CLAUDE.md) before
adding a new one.
