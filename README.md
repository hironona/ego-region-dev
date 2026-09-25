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

The first capture downloads model weights into the HF cache. `steering_boundary` also
fetches `agreeableness.jsonl` from `anthropics/evals` into its `outputs/data/`; both need
network once (see [Cluster runs](#cluster-runs) if the compute node has none).

## Experiments

| folder | question | headline |
|---|---|---|
| `speaker_probe` | is user vs assistant linearly decodable from the residual stream? | ~0.98 balanced accuracy from layer 1; the direction is near-orthogonal to the top PCs |
| `mean_ablation_probe` | which blocks write it? | attention blocks 0–4 |
| `head_ablation_sweep` | which heads? | per-head sweep over that window |
| `steering_boundary` | does trait steering bend near the probe's boundary `w·h + b = 0`? | open — see `docs/HANDOFF.md` |

## Running a pipeline

Every experiment is two processes. `run_capture` loads the model and writes `.npz` under
`experiments/<name>/outputs/`; `analyze` reads only those `.npz` and writes
`outputs/plots/*.png` plus a `results.json`, so plots are free to regenerate and capture is
the only step you pay for twice.

Defaults all live in that experiment's `config.py`, and every default has a matching flag —
`--model`, `--device`, `--seed`, `--out`/`--out-dir` everywhere, plus the per-experiment
knobs below. Flags beat editing `config.py` when you only want one run to differ; each
`config.py` is a private copy, so editing one experiment's never moves another's.

### speaker_probe

```bash
uv run python -m experiments.speaker_probe.run_capture --device cpu
uv run python -m experiments.speaker_probe.analyze
```

Capture: 100 synthetic conversations × 10 turns, 6 evenly spaced content tokens per message
→ one `outputs/capture.npz` (~1.3 GB) with the residual stream at every layer and
speaker/turn/conversation labels. Knobs: `--n`, `--turns`, `--tokens-per-turn`,
`--pool {shared,role}`, `--out`.

Analyze: prints the per-layer accuracy table and writes six plots (accuracy by layer, the
turn-generalisation lines and heatmap, PCA and probe-plane boundaries, PCA-2D vs full)
plus `results.json`. Knobs: `--capture`, `--out-dir`.

`--pool role` is the lexically confounded control, not the real condition. Run it into its
own paths (`--out … --out-dir …`) so it cannot overwrite the `shared` capture.

### mean_ablation_probe

```bash
uv run python -m experiments.mean_ablation_probe.run_capture --device cpu
uv run python -m experiments.mean_ablation_probe.analyze
```

13 conditions — baseline, then a sliding window of 5 consecutive `attn_out` or `mlp_out`
blocks frozen at its context mean — each a separate file in `outputs/captures/`
(`baseline.npz`, `attn_00-04.npz`, …, `mlp_25-27.npz`). Per `config.py` that is ~240 MB
per condition, so budget ~3 GB. Knobs: `--window`, `--components attn mlp`, `--n`,
`--tokens-per-turn`, `--out-dir`.

Analyze reads the whole directory (`--capture-dir`) and writes `accuracy_by_layer.png`,
`delta_heatmap.png`, `final_layer_bars.png`, `results.json`.

`MODEL_NAME`, `N_TURNS`, `SEED` and `POOL` must match `speaker_probe`'s or the accuracies
are not comparable — the dataset builder is shared, so a mismatch silently changes the
conversations rather than erroring.

### head_ablation_sweep

```bash
uv run python -m experiments.head_ablation_sweep.run_capture --device cpu
uv run python -m experiments.head_ablation_sweep.analyze
```

87 conditions: baseline, each of 5×16 individual heads in layers 0–4 mean-ablated at
`hook_z`, each whole layer, and the whole window as the positive control. The config trims
the cost of that many cells to ~10 MB and ~50 s each by using 40 conversations and only
three readout layers (5, 14, 28) — raise `--n` or `--probe-layers` if a result looks
marginal. Knobs: `--sweep-layers`, `--probe-layers`, `--n`, `--tokens-per-turn`,
`--out-dir`.

Analyze writes `head_delta_heatmap.png`, `ranked_heads.png`, `additivity.png`,
`results.json`.

### steering_boundary

Four steps on `Qwen3-8B`, two of which load the model. `analyze` is run twice by design:
the first pass fits the probe and writes the vectors the sweep needs, the second finds the
sweep and plots it.

```bash
uv run python -m experiments.steering_boundary.run_capture --device cuda
uv run python -m experiments.steering_boundary.analyze
uv run python -m experiments.steering_boundary.run_steer --device cuda
uv run python -m experiments.steering_boundary.analyze
```

1. **run_capture** → `capture.npz` (~310 MB): `contrast_X/y` for the steering vector,
   `speaker_X/y` for the boundary, `eval_X` (the unsteered read position of each MCQ) for α*.
   The contrast is `config.VECTOR_METHOD`: `caa` (default) reads each held-out question
   followed by the trait answer vs the other answer, at the answer token, with Yes and No
   balanced across the two classes; `system_prompt` is the old trait-vs-neutral system
   prompt contrast, whose vector steered no better than a random direction on 8B. Knobs:
   `--vector-method`, `--trait`, `--eval-set`, `--n-eval`,
   `--n-vector`, `--n`, `--dtype bfloat16` (fp32 weights alone are 32 GB; bf16 halves that
   and is the only way this fits a 32 GB node).
2. **analyze** → `vectors.npz`, and a printed table of probe accuracy, `cos(w, v)`, median
   α*, `|v|/|h|`, `|v|` and `|h|` per swept layer. α* is in the sweep's units,
   `|c v| / |h|`, where `|h|` is the median norm at the read position. Read the table before
   paying for step 3: if the median α* falls outside `config.STEER_SCALES`, the boundary
   marker lands off-grid. It stops here with a note that `steer.npz` is absent.
3. **run_steer** → `steer.npz`: (trait vector + a random direction of the same norm) ×
   3 layers × 13 scales × 60 eval prompts, ~4.4k forward passes. Each cell records the
   accuracy on each half of the answer key, the Yes-rate and the median `P(Yes)+P(No)`, as
   well as the accuracy. Knobs: `--layers`, `--scales`, `--no-control`, `--n-eval`,
   `--steer-positions {all,last}`, `--dtype`. It asserts the capture's `prompt_style`
   matches before loading the model, so after any prompt change re-run step 1 rather than
   steering a vector fitted elsewhere.
4. **analyze** again → `steering_effect_vs_boundary.png` (split accuracy against the random
   control, with cells that no longer answer greyed out), `steering_validity.png` (Yes-rate
   and answer mass per cell: check this first), `boundary_geometry.png`, `results.json`.

The probe here is refitted on 8B activations, so its accuracy is not comparable with the
0.98 `speaker_probe` reports for 0.6B.

#### Cluster runs

```bash
sbatch slurm/prestage.slurm        # once: 16 GB of weights + the eval jsonl, needs network
sbatch slurm/steering_boundary.slurm
```

The pipeline job runs all four steps offline (`HF_HUB_OFFLINE=1`) and reuses an existing
`capture.npz`. Knobs go through `--export`: `FORCE_CAPTURE=1` (required after a prompt
change), `DTYPE=bfloat16`, `STEER_ARGS="--layers 18 --n-eval 30"`.

## Outputs

Everything lands in `experiments/<name>/outputs/` (gitignored): captures plus `plots/*.png`
and `results.json`. Captures are large — ~310 MB to ~3 GB per experiment, and `outputs/`
has filled the local disk before. Sweeps belong on a GPU or a cluster node.

## What is captured

Per forward pass: `hidden_states` `(L+1, T, D)` (index 0 is the embedding output, then each
layer's `resid_post`), `attentions` `(L, H, T, T)`, `values` `(L, T, D_kv)`, and `logits`
`(T, V)` — selected via `what=`, with optional write hooks for ablation and steering. A
sidecar `.meta.json` records the templated text, token strings and label positions.

## Checks

```bash
uv run python test_pipeline.py
```

12 checks covering npz round-trip, capture shapes, the two Qwen3 `<think>` template
quirks, the lexical-confound control, and that each intervention touches exactly the
tokens it claims to. Several load the model on CPU, so it is not instant.

## More

`docs/HANDOFF.md` — findings, traps, and open items. [CLAUDE.md](CLAUDE.md) — architecture
rules; read before adding an experiment.
