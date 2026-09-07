"""Config for the steering-vs-speaker-boundary experiment. Self-contained by design.

Question: is the speaker probe's decision surface (z = w.h + b = 0, the learned
user/assistant boundary) related to where a trait steering vector starts changing
behaviour? Steering by alpha*v moves z by alpha*(w.v), so each eval prompt has an
exact crossing coefficient alpha* = -z(h) / (w.v). We plot the steering effect
against alpha and mark alpha* on it.

MODEL_NAME / N_TURNS / SEED / POOL must match speaker_probe's for the speaker
boundary here to be the same object as the one measured there.
"""

from pathlib import Path

import numpy as np

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "auto"  # "mps" | "cpu" | "cuda" | "auto"
SEED = 0

# --- speaker boundary (mirrors speaker_probe) ---
N_CONVERSATIONS = 60
N_TURNS = 10
TOKENS_PER_TURN = 2
POOL = "shared"

# --- steering vector (difference-in-means over a system-prompt contrast) ---
TRAIT = "anger"  # key into data.TRAIT_SYSTEM

# --- behavioural eval ---
# anthropics/evals persona set. Yes/No MCQ with a labelled matching answer; we
# score the *trait-consistent* choice, i.e. answer_not_matching_behavior for an
# agreeableness eval steered towards hostility.
EVAL_SET = "agreeableness"
N_EVAL = 150

# --- sweep ---
# Layer index k refers to hidden_states[k]: k=0 is the embedding output, k>=1 is
# the residual stream leaving block k-1. Steering writes to the hook that produces
# exactly that tensor, so probe, vector and intervention all live at the same k.
STEER_LAYERS = (4, 8, 12, 16, 20, 24)
COEFFS = np.linspace(-15, 15, 21)  # wide enough to contain median a* at every swept layer

HERE = Path(__file__).parent
CAPTURE_PATH = HERE / "outputs" / "capture.npz"
VECTOR_PATH = HERE / "outputs" / "vectors.npz"
STEER_PATH = HERE / "outputs" / "steer.npz"
DATA_DIR = HERE / "outputs" / "data"
PLOT_DIR = HERE / "outputs" / "plots"
