"""Config for the per-head ablation sweep over the early attention layers.

The window sweep put the whole speaker signal in attn 0-4. This narrows that to
individual heads: mean-ablate one head's `hook_z` at a time and re-probe.
"""

from pathlib import Path

from core.config import DEVICE, MODEL_NAME  # noqa: F401
from experiments.speaker_probe.config import N_TURNS, POOL, SEED  # noqa: F401

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
