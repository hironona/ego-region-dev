#!/bin/bash
#SBATCH --job-name=steer-boundary
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# 96G, not 32G: Qwen3-8B in float32 is 32 GB of weights, and loading transiently
# holds the HF state dict alongside the HookedTransformer copy. --dtype bfloat16
# halves both and fits in 48G, at the cost of no longer matching the float32
# numerics every earlier capture used -- so it is a flag, not the default.
#SBATCH --mem=96G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

# Full steering_boundary pipeline, four steps, one job:
#
#   1. run_capture   loads the model, writes capture.npz (~310 MB at 8B)
#   2. analyze       no model; fits the probe, writes vectors.npz, prints the
#                    layer table (median a*, |v|/|h|) and stops -- steer.npz absent
#   3. run_steer     loads the model; 2 kinds (trait + random control) x
#                    3 layers x 12 non-zero scales x 60 prompts + 60, ~4.4k
#                    forward passes (the c=0 column is shared by every layer
#                    and both kinds, so it is one pass per prompt)
#   4. analyze       same command as 2; now finds steer.npz and writes plots
#                    plus results.json
#
# Step 2 is deliberately run twice. That is the documented design, not a slip.
#
# Submit with:  sbatch slurm/steering_boundary.slurm
# after slurm/prestage.slurm has completed -- this job runs offline and will
# fail on a cache miss otherwise.
#
# Knobs, all optional, set on the sbatch line as --export:
#   FORCE_CAPTURE=1        redo step 1 even if capture.npz exists. Needed after
#                          any change to the prompt or to config.VECTOR_METHOD:
#                          step 2 refuses a capture built with another vector
#                          method, and step 3 one whose prompt_style does not
#                          match, rather than steering what they were not fitted on.
#   DTYPE=bfloat16         halve the memory and roughly halve the time, at the
#                          cost of float32 reference numerics
#   STEER_ARGS="..."       extra args for run_steer, e.g. a smaller grid:
#                          STEER_ARGS="--layers 18 --scales -1 -0.5 0 0.5 1"
#                          or no random control:  STEER_ARGS="--no-control"
#                          or fewer eval items:  STEER_ARGS="--n-eval 30"
#
# deigo is CPU-only. --device cpu is passed explicitly rather than left on
# "auto": core.model.resolve_device tries MPS first, and TransformerLens warns
# the MPS backend can be silently wrong, so nothing here should ever resolve
# to it by accident. An 8B model is where a GPU partition starts to be worth
# the queue: the sweep is ~2.6k forward passes and each is a second or more of
# CPU. Nothing below assumes CPU except the --device flag.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

OUT="$REPO/experiments/steering_boundary/outputs"
PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || { echo "no venv at $PY -- run 'uv sync' on the login node"; exit 1; }

# .venv/bin/python directly, not `uv run`: uv would try to re-resolve the
# environment, and this node has no network.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Everything was fetched by slurm/prestage.slurm. If that was skipped this
# fails immediately with a cache miss instead of stalling on an unroutable
# request.
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

step() { echo; echo "=== $* === $(date '+%F %T')"; }

echo "job $SLURM_JOB_ID on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus"
DTYPE="${DTYPE:-}"
[[ -n "$DTYPE" ]] && DTYPE_ARG=(--dtype "$DTYPE") || DTYPE_ARG=()

# The grid is non-uniform and in units of |h| (|c v| / |h|), so print it rather
# than a min/max/step that would misdescribe it.
"$PY" -c "from experiments.steering_boundary import config as c; \
print('model: ', c.MODEL_NAME, '| dtype', c.DTYPE); \
print('layers:', c.STEER_LAYERS); \
print('vector:', c.VECTOR_METHOD, '| random control', c.RANDOM_CONTROL); \
print('scales:', [round(float(x), 2) for x in c.STEER_SCALES]); \
print('cells: ', (1 + c.RANDOM_CONTROL) * len(c.STEER_LAYERS) * (len(c.STEER_SCALES) - 1) * c.N_EVAL + c.N_EVAL, \
'forward passes over', c.N_EVAL, 'eval prompts')"

if [[ -f "$OUT/capture.npz" && "${FORCE_CAPTURE:-0}" != "1" ]]; then
    step "1/4 capture -- reusing $OUT/capture.npz (FORCE_CAPTURE=1 to redo)"
else
    step "1/4 capture"
    "$PY" -m experiments.steering_boundary.run_capture --device cpu ${DTYPE_ARG[@]+"${DTYPE_ARG[@]}"}
    # A new capture invalidates both downstream files and nothing in the code
    # checks that, so drop the stale sweep rather than let step 2 plot it.
    rm -f "$OUT/steer.npz" "$OUT/steer.meta.json"
fi

step "2/4 analyze -- probe, vectors, layer table"
"$PY" -m experiments.steering_boundary.analyze

step "3/4 steer -- the expensive one"
"$PY" -m experiments.steering_boundary.run_steer --device cpu ${DTYPE_ARG[@]+"${DTYPE_ARG[@]}"} ${STEER_ARGS:-}

step "4/4 analyze -- plots + results.json"
"$PY" -m experiments.steering_boundary.analyze

step "done"
ls -la "$OUT" "$OUT/plots"
