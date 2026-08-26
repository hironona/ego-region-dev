"""Config for the self/user activation-geometry experiment."""

from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "auto"  # "mps" | "cpu" | "cuda" | "auto"

PROMPT = "Who are you, and who am I?"
SYSTEM = None  # optional system message

HERE = Path(__file__).parent
CAPTURE_PATH = HERE / "outputs" / "capture.npz"
PLOT_DIR = HERE / "outputs" / "plots"
