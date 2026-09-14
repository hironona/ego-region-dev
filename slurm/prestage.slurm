#!/bin/bash
#SBATCH --job-name=steer-prestage
#SBATCH --partition=datacp
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

# One-time download step for the steering_boundary pipeline. Run before
# slurm/steering_boundary.slurm:
#
#   sbatch slurm/prestage.slurm
#
# Fetches two things, neither of which is cached anywhere yet:
#   Qwen/Qwen3-0.6B          from huggingface.co  -> $HF_HOME
#   agreeableness.jsonl      from raw.githubusercontent.com
#                            -> experiments/steering_boundary/outputs/data/
#
# Both land on the shared /home filesystem, so the compute node that runs the
# actual sweep sees them and can run with HF_HUB_OFFLINE=1.
#
# Why datacp and not compute: this job is pure network I/O, and deigo-mover is
# the partition provisioned for transfers in and out of the cluster. Ordinary
# compute nodes are the ones least likely to have a route out, which is the
# entire reason this step is separate from the pipeline job.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
# The opposite of the pipeline job: this is the one step that must reach the
# network, so make sure no inherited offline flag silently turns it into a
# cache-only no-op that "succeeds" and leaves the sweep to fail later.
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || { echo "no venv at $PY -- run 'uv sync' on the login node"; exit 1; }

echo "job ${SLURM_JOB_ID:-?} on $(hostname)"

# Fail with a readable message rather than a stack trace if this partition
# turns out to have no route out either.
echo "=== connectivity ==="
for host in huggingface.co raw.githubusercontent.com; do
    if curl -sSf --max-time 30 -o /dev/null "https://$host"; then
        echo "  ok    $host"
    else
        echo "  FAIL  $host -- no route out from $(hostname)."
        echo "        Try a different partition, or run this on the login node."
        exit 1
    fi
done

echo
echo "=== model -> $HF_HOME ==="
"$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
from experiments.steering_boundary import config
print("cached:", snapshot_download(config.MODEL_NAME))
PYEOF

echo
echo "=== eval set -> experiments/steering_boundary/outputs/data ==="
"$PY" - <<'PYEOF'
from experiments.steering_boundary import config
from experiments.steering_boundary.data import eval_questions
# Two calls, because the sweep and the vector read disjoint slices and both
# must exist: rows [0:N_EVAL] are scored, [VECTOR_OFFSET:+N_VECTOR] elicit.
eval_questions(config.EVAL_SET, config.N_EVAL, config.DATA_DIR)
eval_questions(config.EVAL_SET, config.N_VECTOR, config.DATA_DIR, offset=config.VECTOR_OFFSET)
print("cached:", config.DATA_DIR / f"{config.EVAL_SET}.jsonl")
PYEOF

# Prove the cache is actually usable offline, here, where a failure is cheap --
# rather than 12 hours into a queue on a node that cannot re-download.
echo
echo "=== offline readback ==="
HF_HUB_OFFLINE=1 "$PY" - <<'PYEOF'
from transformers import AutoTokenizer
from experiments.steering_boundary import config
tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
print("tokenizer loads offline, vocab", len(tok))
PYEOF

echo
echo "=== prestage done ==="
du -sh "$HF_HOME" "$REPO/experiments/steering_boundary/outputs/data"
