"""Config for the mean-ablation speaker-probe experiment. Self-contained by design.

Localises where the user/assistant signal is written into the residual stream by
mean-ablating a sliding window of attention or MLP blocks and re-probing.

Every experiment keeps its own copy of these values rather than importing them
from a shared module, so changing one experiment can never silently move another.
The flip side: MODEL_NAME, N_TURNS, SEED and POOL must match speaker_probe's for
the accuracies to be comparable across the two experiments — the dataset builder
is shared, so a mismatch here silently changes the conversations.
"""

from pathlib import Path

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # 32 blocks, 32 heads, d_model 4096
DEVICE = "auto"  # "mps" | "cpu" | "cuda" | "auto"

N_CONVERSATIONS = 100
N_TURNS = 10
SEED = 0
POOL = "shared"

# Conditions x this many rows x (L+1) layers x d_model dims of fp16 is the whole
# disk cost of the experiment: 2 tokens/turn -> ~4k rows. On Qwen3-0.6B (13
# conditions, 29 x 1024) that was ~240 MB per condition; on Llama-3.1-8B (15
# conditions, 33 x 4096) it is ~1.1 GB per condition, ~16 GB in all.
# Raise it if a probe ever looks sample-starved; the effect here is not subtle.
TOKENS_PER_TURN = 2

WINDOW = 5  # consecutive blocks frozen per condition
COMPONENTS = ("attn", "mlp")

HERE = Path(__file__).parent
CAPTURE_DIR = HERE / "outputs" / "captures"
PLOT_DIR = HERE / "outputs" / "plots"
