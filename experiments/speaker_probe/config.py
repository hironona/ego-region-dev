"""Config for the speaker-probe experiment. Self-contained by design.

Every experiment keeps its own copy of these values rather than importing them
from a shared module, so changing one experiment can never silently move another.
"""

from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "auto"  # "mps" | "cpu" | "cuda" | "auto"

N_CONVERSATIONS = 100
N_TURNS = 10
TOKENS_PER_TURN = 6
SEED = 0
POOL = "shared"  # "shared" = both roles draw the same sentences; "role" = the lexically confounded baseline

HERE = Path(__file__).parent
CAPTURE_PATH = HERE / "outputs" / "capture.npz"
PLOT_DIR = HERE / "outputs" / "plots"
