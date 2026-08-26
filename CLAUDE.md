# Architecture conventions

Two layers, one rule: **`core/` never knows about an experiment.**

```
core/                        reusable, experiment-agnostic
  model.py                   load model + tokenizer, device resolution (mps/cpu/cuda/auto)
  capture.py                 forward hooks, activation/attention capture, npz storage
experiments/<name>/          one folder per experiment, built on core/
  config.py                  model, prompt, output paths
  run_capture.py             prompt -> chat template -> forward pass -> outputs/capture.npz
  analyze.py                 capture.npz -> outputs/plots/*.png (never loads the model)
  outputs/                   capture + plots (gitignored)
```

## Adding an experiment

Make `experiments/<new_name>/` with the same three modules. Do not edit `core/` to
make it fit — if `core/` genuinely lacks a capability (a new hook target, a new
storage format), add it there in a generic form that any experiment could use, and
keep prompt/role/analysis choices in the experiment.

## Rules that matter

- **Capture and analysis are separate processes.** `run_capture.py` is the only thing
  that loads the model; `analyze.py` reads `capture.npz` only, so plots can be
  regenerated without a forward pass. Keep it that way.
- **Verify tokenizer facts, don't assume them.** Role-token positions are found by
  locating the turn-start special token and reading what actually follows it
  (`find_role_token_positions`), not by hardcoding a token string.
- **Capture goes through TransformerLens.** `core.model.load` returns a
  `HookedTransformer`, and `core.capture.run` reads named hooks
  (`blocks.{l}.attn.hook_v`, `hook_pattern`, `hook_resid_pre`) out of
  `run_with_cache`. Load with `from_pretrained_no_processing` — LayerNorm folding
  and weight centring would change the activations you are looking at.
- Raw captures are the durable artifact. Anything derived (PCA, plots, stats) belongs
  in `analyze.py` and should be cheap to throw away and recompute.
