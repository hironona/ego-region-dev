# Plan: speaker-probe

Goal: A new `experiments/speaker_probe/` pipeline that captures per-token residual activations from 100 preset 10-turn conversations, labels each token user/assistant, and reports layerwise logistic-regression probe accuracy both turn-agnostic (Exp 1) and per-turn-pair (Exp 2), with plots.
Created: 2026-08-26  ·  Orchestrator model: opus-5

## Constraints

- `core/` stays experiment-agnostic. The only permitted core changes are the two generic
  ones in Step 1. No conversation/label/probe logic in `core/`.
- Capture goes through `core.capture` and `core.model.load` (TransformerLens,
  `from_pretrained_no_processing`). Do not call HF transformers directly.
- Capture and analysis stay separate processes: `run_capture.py` is the only module that
  loads the model; `analyze.py` reads `capture.npz` only.
- `experiments/self_user_geometry/` is read-only for every step.
- Role/turn boundaries are derived from the tokenizer, never hardcoded token strings.
- **No `<think>` block may appear anywhere in the templated text.** Verified during recon:
  Qwen3's chat template injects `<think>\n\n</think>\n\n` into the *last* assistant
  message of the history, and `enable_thinking=False` does not remove it (it only adds one
  to the generation prompt as well). The sanctioned workaround is in Step 3.
- Config inherits `MODEL_NAME` / `DEVICE` by importing them from
  `experiments.self_user_geometry.config`.

## Open questions (assumed, flag in Outcome if wrong)

- **Which tokens get a label.** Assumed: *content tokens only*. The chat-template header
  (`<|im_start|>user`) and `<|im_end|>` are excluded, because the literal role token would
  make the probe trivially perfect and measure the template, not the geometry.
- **Storage budget.** Every token of even 100 convs across 29 layers at fp32 is several GB.
  Assumed: subsample `TOKENS_PER_TURN` (default 6) content tokens per message, store fp16,
  uncompressed. 100 x 20 x 6 = ~12k rows -> ~700 MB, ~2.4k rows per Exp-2 set.
- **"Preset" conversations.** Assumed: synthetically templated from word banks with a
  fixed seed — no model generation, one forward pass per conversation.
- **Conversation count.** 100 (user's instruction, down from 500) to keep this test cheap.
  `N_CONVERSATIONS` is the single lever to scale it back up.
- **Turn definition.** Turn k = the (user_k, assistant_k) pair, k in 0..9. Exp 2 splits on
  pairs {0,1},{2,3},{4,5},{6,7},{8,9}.

## Steps

### Step 1 — Make `core.capture` able to capture cheaply
- **Files (writable):** `core/capture.py`
- **Files (read-only):** `experiments/self_user_geometry/run_capture.py`
- **Change:** Add two backward-compatible generic parameters, nothing else:
  1. `run(model, input_ids, device, what=("hidden_states", "attentions", "values"))` —
     only the requested hook families are registered and returned. Existing callers that
     omit `what` get today's exact behaviour and dict keys.
  2. `save(path, arrays, meta, compress=True)` — `compress=False` uses `np.savez`.
     Default keeps current behaviour.
- **Verify:** `uv run python -c "
import torch,numpy as np;from core import capture,model as M
m,tok,dev=M.load('Qwen/Qwen3-0.6B','auto')
ids=tok('hello there',return_tensors='pt')['input_ids']
a=capture.run(m,ids,dev,what=('hidden_states',));assert set(a)=={'hidden_states'},a.keys()
b=capture.run(m,ids,dev);assert set(b)=={'hidden_states','attentions','values'}
assert np.allclose(a['hidden_states'],b['hidden_states']);print('ok',a['hidden_states'].shape)"`
- **Depends on:** none
- **Group:** A

### Step 2 — Config + conversation generator
- **Files (writable):** `experiments/speaker_probe/__init__.py`,
  `experiments/speaker_probe/config.py`, `experiments/speaker_probe/conversations.py`
- **Files (read-only):** `experiments/self_user_geometry/config.py`
- **Change:**
  - `config.py`: `from experiments.self_user_geometry.config import MODEL_NAME, DEVICE`
    (re-exported), plus `N_CONVERSATIONS=100`, `N_TURNS=10`, `TOKENS_PER_TURN=6`,
    `SEED=0`, `CAPTURE_PATH = HERE/'outputs'/'capture.npz'`, `PLOT_DIR = HERE/'outputs'/'plots'`.
  - `conversations.py`: `build(n, n_turns, seed) -> list[list[dict]]` returning
    OpenAI-style `[{"role":"user"|"assistant","content":str}, ...]` of length `2*n_turns`,
    strictly alternating and starting with `user`. Content comes from small topic/word
    banks combined with `random.Random(seed)`; deterministic for a given seed. Keep it
    short — a couple of templates per role is enough, do not build a DSL.
- **Verify:** `uv run python -c "
from experiments.speaker_probe import config,conversations as C
a=C.build(100,10,0);b=C.build(100,10,0)
assert config.N_CONVERSATIONS==100 and len(a)==100 and all(len(c)==20 for c in a)
assert all(m['role']==('user' if i%2==0 else 'assistant') for c in a for i,m in enumerate(c))
assert [m['content'] for m in a[7]]==[m['content'] for m in b[7]]
assert len({c[0]['content'] for c in a})>10
print(config.MODEL_NAME, config.DEVICE, a[0][0]['content'][:60])"`
- **Depends on:** none
- **Group:** A

### Step 3 — Capture with speaker labels
- **Files (writable):** `experiments/speaker_probe/run_capture.py`
- **Files (read-only):** everything else
- **Change:** For each conversation, build the templated text **think-free**: call
  `apply_chat_template(messages + [{"role":"user","content":""}],
  add_generation_prompt=False, tokenize=False)` and truncate at the last
  `<|im_start|>user`. Recon verified the injection only ever lands on the final assistant
  message, so the truncated string contains no `<think>`; **assert `"<think>" not in text`**
  and fail loudly if a future template version changes this.
  Recover each message's token span by tokenizing the per-message blocks
  `f"<|im_start|>{role}\n{content}<|im_end|>\n"` independently and taking cumulative
  offsets; recon verified this concatenation is token-exact against the full text, so
  **assert the concatenated ids equal the full ids** rather than trusting it. (Prefix-diff
  tokenization is *not* usable here — the think block moves between prefixes.)
  Within each message span, drop the leading role-header tokens and the trailing
  `<|im_end|>`/newline specials (locate `<|im_start|>` / `<|im_end|>` via
  `convert_tokens_to_ids`, assert they are found), then take up to `TOKENS_PER_TURN`
  evenly-spaced content positions. Run `capture.run(..., what=("hidden_states",))` once per
  conversation, gather those positions into rows.
  Save via `capture.save(..., compress=False)`:
  - `X` float16 `(N, L+1, D)`, `speaker` uint8 `(N,)` (0=user, 1=assistant),
    `turn` uint8 `(N,)` (0..9), `conv` int32 `(N,)`.
  - meta: model, device, counts, `n_layers`, `d_model`, label convention, seed, and one
    example `templated_text` (which a reader can eyeball for absence of `<think>`).
  CLI: `--n`, `--turns`, `--tokens-per-turn`, `--model`, `--device`, `--out`, defaulting
  from config. Print progress every 50 conversations.
- **Verify (smoke, small n — the full run is Step 5):** `uv run python -m experiments.speaker_probe.run_capture --n 4 --out /tmp/sp_smoke.npz && uv run python -c "
from core import capture;import numpy as np
a,m=capture.load('/tmp/sp_smoke.npz')
X,s,t,c=a['X'],a['speaker'],a['turn'],a['conv']
assert X.ndim==3 and X.dtype==np.float16, (X.shape,X.dtype)
assert len(X)==len(s)==len(t)==len(c) and set(np.unique(s))=={0,1}
assert set(np.unique(t))==set(range(10)) and set(np.unique(c))=={0,1,2,3}
assert np.isfinite(X.astype(np.float32)).all()
assert '<think>' not in m['templated_text']
print(X.shape, m['label_convention'])"`
- **Depends on:** Steps 1, 2
- **Group:** B

### Step 4 — Analysis: both experiments + plots
- **Files (writable):** `experiments/speaker_probe/analyze.py`
- **Files (read-only):** `experiments/speaker_probe/config.py`, `core/capture.py`
- **Change:** Load `capture.npz` only — never the model. Use
  `sklearn.linear_model.LogisticRegression` inside a `Pipeline` with `StandardScaler`
  (`max_iter=1000`). Split **by conversation id**, not by row, so tokens from one
  conversation cannot straddle train/test (`GroupShuffleSplit` or a manual conv-id split,
  test 25%). Report balanced accuracy (classes are balanced by construction, but say so).
  - **Exp 1 (turn-agnostic):** one probe per layer `0..L` on all rows.
    Plot `exp1_accuracy_by_layer.png`: accuracy vs layer, chance line at 0.5.
  - **Exp 2 (turn-sensitive):** for each layer, 5 probes on the turn-pair subsets
    {0,1},{2,3},{4,5},{6,7},{8,9}, trained and tested within their own subset (same
    conversation-level split). Plot `exp2_accuracy_by_layer_turnset.png` (5 lines over
    layers) and `exp2_heatmap.png` (layer x turn-set accuracy heatmap).
  - Write `outputs/results.json` with every accuracy, and print a compact table.
  - `--capture`, `--out-dir` CLI args; `matplotlib.use("Agg")`.
- **Verify:** `uv run python -m experiments.speaker_probe.analyze --capture /tmp/sp_smoke.npz --out-dir /tmp/sp_plots && ls /tmp/sp_plots/*.png && uv run python -c "
import json;r=json.load(open('/tmp/sp_plots/results.json'));print(sorted(r))"`
  (A 4-conversation smoke set may probe near chance — this step verifies it *runs and
  writes*, not that the numbers are good.)
- **Depends on:** Step 3
- **Group:** C

### Step 5 — Full run + report
- **Files (writable):** `experiments/speaker_probe/outputs/**` (gitignored artifacts only)
- **Files (read-only):** all source
- **Change:** Run the real thing: `run_capture` with the config defaults (100 x 10 turns),
  then `analyze`. No source edits. If a source bug surfaces, stop and report it rather
  than patching outside the boundary.
- **Verify:** `uv run python -m experiments.speaker_probe.run_capture && uv run python -m experiments.speaker_probe.analyze && uv run python -c "
import json;r=json.load(open('experiments/speaker_probe/outputs/plots/results.json'));print(json.dumps(r,indent=1)[:800])"`
- **Depends on:** Step 4
- **Group:** D

## Parallel groups

- Group A: Steps 1, 2 — disjoint writable sets (`core/capture.py` vs the new experiment
  folder); Step 2 does not import the changed core API.
- Group B: Step 3 — needs both A steps landed and verified.
- Group C: Step 4 — needs a real `capture.npz` schema to read.
- Group D: Step 5 — needs everything.

## Outcome

<!-- filled in at stage 6 -->
