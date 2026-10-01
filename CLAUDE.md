# Architecture conventions

Two layers, one rule: **`core/` never knows about an experiment.**

```
core/                        reusable, experiment-agnostic
  model.py                   load(): HookedTransformer + tokenizer, device resolution (mps/cpu/cuda/auto)
  chat.py                    per-family chat formats (llama3, qwen3), picked from the tokenizer;
                             encode(): templated ids + per-message token spans, verified against
                             apply_chat_template; prompt(): a prompt ending at the answer position
  capture.py                 run(): forward pass with named hooks + optional write
                             interventions; save()/load(): npz + sidecar .meta.json
experiments/<name>/          one folder per experiment, built on core/
  config.py                  model, sizes, seeds, output paths — no imports from other experiments
  run_capture.py             prompts -> forward pass -> outputs/capture.npz
  analyze.py                 capture.npz -> outputs/plots/*.png (never loads the model)
  outputs/                   captures + plots (gitignored)
```

Every pipeline defaults to `meta-llama/Llama-3.1-8B-Instruct`. The results so far were
measured on Qwen3 (`speaker_probe`/ablations on 0.6B, `steering_boundary`/`mirror_attribution`
on 8B); the Qwen3 path is kept and runs with `--model Qwen/Qwen3-...`.

Current experiments, in the order they were built:

| experiment | question | extra modules |
|---|---|---|
| `speaker_probe` | is user vs assistant linearly decodable from the residual stream? (~0.98 bal. acc from L1 on Qwen3-0.6B) | `conversations.py` — the shared synthetic-dialogue builder |
| `mean_ablation_probe` | which blocks write that signal? mean-ablate a sliding attn/MLP window and re-probe | — |
| `head_ablation_sweep` | narrows the implicated window to individual heads (`hook_z`) | — |
| `steering_boundary` | does a trait steering vector start changing behaviour near the probe's boundary? | `data.py` (CAA answer pairs, legacy trait prompts, anthropics/evals), `run_steer.py` (second model-loading step) |
| `mirror_attribution` | does reflecting the history across the self vector's boundary (`v = μ_asst − μ_user`, fitted on speaker_probe's conversations) make the model misattribute who said what? | `opinions.py` (n-turn opinion conversations + "who said it, the user or the assistant?"), `run_mirror.py` (second model-loading step: reflection hook + forced-choice User/Assistant readout) |

`docs/HANDOFF.md` is the running research log: what has actually been measured, what is
still unverified, and the open items. Read it before drawing conclusions from any plot.

## Adding an experiment

Make `experiments/<new_name>/` with the same three modules. Do not edit `core/` to
make it fit — if `core/` genuinely lacks a capability (a new hook target, a new
storage format), add it there in a generic form that any experiment could use, and
keep prompt/role/analysis choices in the experiment.

## Rules that matter

- **Capture and analysis are separate processes.** `run_capture.py` (and
  `steering_boundary/run_steer.py`, `mirror_attribution/run_mirror.py`) are the only
  things that load the model;
  `analyze.py` reads the `.npz` only, so plots can be regenerated without a forward
  pass. Keep it that way.
- **`config.py` is copied, not shared.** Every experiment holds its own `MODEL_NAME`,
  `N_TURNS`, `SEED`, `POOL`, so editing one experiment can never silently move
  another. The flip side: those four must *match* `speaker_probe`'s for accuracies to
  be comparable across experiments, because the dataset builder is shared.
- **Code may be reused across experiments; config may not.** Later experiments import
  `speaker_probe.conversations.build`, `speaker_probe.run_capture.build_input`, and
  `speaker_probe.analyze.{probe_accuracy, conv_train_test_masks}` (and `mirror_attribution` reuses
  `steering_boundary.analyze.fit_speaker_boundary` for the raw-space probe) — that is deliberate,
  so "the probe" means one implementation. `speaker_probe` itself imports from nobody.
- **One layer index, everywhere: `hidden_states[k]`.** `k=0` is the embedding output,
  `k>=1` is the residual stream leaving block `k-1` (last row is pre-final-norm). Probe,
  steering vector and intervention hook must all refer to the same `k` — see
  `run_steer.hook_name(k)`.
- **Verify tokenizer facts, don't assume them — then check what they do to the
  model.** All template handling lives in `core/chat.py`; never hardcode a role header
  in an experiment. Llama 3.1's template has no thinking mode, but it always prepends
  BOS and a system block (`Cutting Knowledge Date … Today Date: 26 Jul 2024`) even when
  there is no system message, merges a caller's system prompt into that same block, and
  trims message content. `ChatFormat.preamble` pins the first and the tests pin the
  rest; the preamble is never part of any message's span or role mask.
  Qwen3's chat template has two opposite `<think>` behaviours: in conversation
  *history* it injects an empty block into the last assistant message and
  `enable_thinking=False` does not remove it (worked around by appending an empty user
  turn and truncating); in a *generation prompt* the default is clean and
  `enable_thinking=False` *adds* the block. Both are pinned by tests — if a template
  update breaks them, fix the code, not the test.
  The second fact is a trap, and the eval fell into it for a while: Qwen3 thinks by
  default, so the *clean* generation prompt is the one the model answers by emitting
  `<think>` and reasoning for hundreds of tokens. Reading answer logits there samples
  the tail of the distribution (Yes/No near rank 100k, ~0 probability) and yields a
  flat 0.5 that is indistinguishable from a null result. `enable_thinking=False` is
  what puts the answer at the read position. Assert on the model's distribution, not
  on the prompt string — a clean string is not a think-free model.
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

- Captures are large (~310 MB for `steering_boundary` on Qwen3-8B, ~1.3 GB for
  `speaker_probe` on Qwen3-0.6B — ~4.5x that per row at Llama-3.1-8B's 33 x 4096);
  `outputs/` is gitignored and has filled the local disk before. Sweeps belong on a GPU.
- TransformerLens warns that **the MPS backend can be silently wrong** (torch 2.13).
  `--device mps` is fine for smoke tests, not for anything load-bearing.
- `test_pipeline.py` is the whole test suite — plain asserts, no framework. Several
  checks load the default model on CPU (once, cached; 8B in fp32 is ~32 GB of RAM).
  Tokenizer-only checks run on both the Llama and Qwen3 tokenizers.
  Llama is gated on the Hub: the account behind the HF token must have accepted
  Meta's licence. Add one check per non-obvious invariant, not per function.
