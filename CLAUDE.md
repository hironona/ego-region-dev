# Architecture conventions

Two layers, one rule: **`core/` never knows about an experiment.**

```
core/                        reusable, experiment-agnostic
  model.py                   load(): HookedTransformer + tokenizer, device resolution (mps/cpu/cuda/auto)
  capture.py                 run(): forward pass with named hooks + optional write
                             interventions; save()/load(): npz + sidecar .meta.json
experiments/<name>/          one folder per experiment, built on core/
  config.py                  model, sizes, seeds, output paths — no imports from other experiments
  run_capture.py             prompts -> forward pass -> outputs/capture.npz
  analyze.py                 capture.npz -> outputs/plots/*.png (never loads the model)
  outputs/                   captures + plots (gitignored)
```

Current experiments, in the order they were built:

| experiment | question | extra modules |
|---|---|---|
| `speaker_probe` | is user vs assistant linearly decodable from the residual stream? (~0.98 bal. acc from L1) | `conversations.py` — the shared synthetic-dialogue builder |
| `mean_ablation_probe` | which blocks write that signal? mean-ablate a sliding attn/MLP window and re-probe | — |
| `head_ablation_sweep` | narrows the implicated window to individual heads (`hook_z`) | — |
| `steering_boundary` | does a trait steering vector start changing behaviour near the probe's boundary? | `data.py` (trait prompts + anthropics/evals), `run_steer.py` (second model-loading step) |

`docs/HANDOFF.md` is the running research log: what has actually been measured, what is
still unverified, and the open items. Read it before drawing conclusions from any plot.

## Adding an experiment

Make `experiments/<new_name>/` with the same three modules. Do not edit `core/` to
make it fit — if `core/` genuinely lacks a capability (a new hook target, a new
storage format), add it there in a generic form that any experiment could use, and
keep prompt/role/analysis choices in the experiment.

## Rules that matter

- **Capture and analysis are separate processes.** `run_capture.py` (and
  `steering_boundary/run_steer.py`) are the only things that load the model;
  `analyze.py` reads the `.npz` only, so plots can be regenerated without a forward
  pass. Keep it that way.
- **`config.py` is copied, not shared.** Every experiment holds its own `MODEL_NAME`,
  `N_TURNS`, `SEED`, `POOL`, so editing one experiment can never silently move
  another. The flip side: those four must *match* `speaker_probe`'s for accuracies to
  be comparable across experiments, because the dataset builder is shared.
- **Code may be reused across experiments; config may not.** Later experiments import
  `speaker_probe.conversations.build`, `speaker_probe.run_capture.build_input`, and
  `speaker_probe.analyze.{probe_accuracy, conv_train_test_masks}` — that is deliberate,
  so "the probe" means one implementation. `speaker_probe` itself imports from nobody.
- **One layer index, everywhere: `hidden_states[k]`.** `k=0` is the embedding output,
  `k>=1` is the residual stream leaving block `k-1` (last row is pre-final-norm). Probe,
  steering vector and intervention hook must all refer to the same `k` — see
  `run_steer.hook_name(k)`.
- **Verify tokenizer facts, don't assume them.** Qwen3's chat template has two opposite
  `<think>` behaviours: in conversation *history* it injects an empty block into the last
  assistant message and `enable_thinking=False` does not remove it (worked around by
  appending an empty user turn and truncating); in a *generation prompt* the default is
  clean and `enable_thinking=False` *adds* the block. Both are pinned by tests — if a
  template update breaks them, fix the code, not the test.
- **Capture goes through TransformerLens.** `core.model.load` returns a
  `HookedTransformer`, and `core.capture.run` reads named hooks
  (`blocks.0.hook_resid_pre` + `blocks.{l}.hook_resid_post`, `attn.hook_pattern`,
  `attn.hook_v`) out of
  `run_with_cache`. Load with `from_pretrained_no_processing` — LayerNorm folding
  and weight centring would change the activations you are looking at.
- **Anything above chance at layer 0 is content, not representation.** The `role` pool
  is kept as the deliberately confounded baseline; `shared` is the real one.
- Raw captures are the durable artifact. Anything derived (probes, PCA, plots, stats)
  belongs in `analyze.py` and should be cheap to throw away and recompute.

## Operational notes

- Captures are large (~160 MB for `steering_boundary`, ~1.3 GB for `speaker_probe`);
  `outputs/` is gitignored and has filled the local disk before. Sweeps belong on a GPU.
- TransformerLens warns that **the MPS backend can be silently wrong** (torch 2.13).
  `--device mps` is fine for smoke tests, not for anything load-bearing.
- `test_pipeline.py` is the whole test suite — plain asserts, no framework. Several
  checks load the model on CPU. Add one check per non-obvious invariant, not per function.
