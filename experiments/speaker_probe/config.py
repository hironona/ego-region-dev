"""Config for the speaker-probe experiment."""

from pathlib import Path

from core.config import MODEL_NAME, DEVICE  # noqa: F401

N_CONVERSATIONS = 100
N_TURNS = 10
TOKENS_PER_TURN = 6
SEED = 0
POOL = "shared"  # "shared" = both roles draw the same sentences; "role" = the lexically confounded baseline

HERE = Path(__file__).parent
CAPTURE_PATH = HERE / "outputs" / "capture.npz"
PLOT_DIR = HERE / "outputs" / "plots"
