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

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "auto"  # "mps" | "cpu" | "cuda" | "auto"

N_CONVERSATIONS = 100
N_TURNS = 10
SEED = 0
POOL = "shared"

# 13 conditions x this many rows x 29 layers x 1024 dims of fp16 is the whole disk
# cost of the experiment: 2 tokens/turn -> ~4k rows -> ~240 MB per condition.
# Raise it if a probe ever looks sample-starved; the effect here is not subtle.
TOKENS_PER_TURN = 2

WINDOW = 5  # consecutive blocks frozen per condition
COMPONENTS = ("attn", "mlp")

HERE = Path(__file__).parent
CAPTURE_DIR = HERE / "outputs" / "captures"
PLOT_DIR = HERE / "outputs" / "plots"
