"""Config for the per-head ablation sweep. Self-contained by design.

The window sweep put the whole speaker signal in attn 0-4. This narrows that to
individual heads: mean-ablate one head's `hook_z` at a time and re-probe.

Every experiment keeps its own copy of these values rather than importing them
from a shared module, so changing one experiment can never silently move another.
The flip side: MODEL_NAME, N_TURNS, SEED and POOL must match speaker_probe's for
the accuracies to be comparable across experiments — the dataset builder is
shared, so a mismatch here silently changes the conversations.
"""

from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "auto"  # "mps" | "cpu" | "cuda" | "auto"

N_TURNS = 10
SEED = 0
POOL = "shared"

SWEEP_LAYERS = (0, 1, 2, 3, 4)  # the window the previous experiment implicated

# 5 layers x 16 heads = 80 head conditions, plus per-layer and whole-window
# references and a baseline. Two knobs keep that affordable, and both are safe
# because the effect being measured is ~0.45 accuracy, not a few points:
#   - 40 conversations (1600 rows) instead of 100
#   - only three readout layers instead of all 29
# ~10 MB and ~50 s per condition. Raise either if a result looks marginal.
N_CONVERSATIONS = 40
TOKENS_PER_TURN = 2
PROBE_LAYERS = (5, 14, 28)  # just past the window, mid-stack, final

HERE = Path(__file__).parent
CAPTURE_DIR = HERE / "outputs" / "captures"
PLOT_DIR = HERE / "outputs" / "plots"
