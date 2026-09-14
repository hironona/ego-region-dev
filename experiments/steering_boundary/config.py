"""Config for the steering-vs-speaker-boundary experiment. Self-contained by design.

Question: is the speaker probe's decision surface (z = w.h + b = 0, the learned
user/assistant boundary) related to where a trait steering vector starts changing
behaviour? Steering by alpha*v moves z by alpha*(w.v), so each eval prompt has an
exact crossing coefficient alpha* = -z(h) / (w.v). We plot the steering effect
against alpha and mark alpha* on it.

N_TURNS / SEED / POOL match speaker_probe's, so the speaker boundary fitted
here is built the same way as the one measured there. MODEL_NAME deliberately
does not: speaker_probe ran on Qwen3-0.6B, and at that size the unsteered model
answers the agreeableness eval at chance (it picks one of Yes/No largely
regardless of the question), so a steering sweep on top of it has no behaviour
to move. run_steer prints the unsteered Yes-rate for exactly this reason. The
probe here is therefore refitted on 8B activations from this experiment's own
capture -- its accuracy is a number about 8B, not comparable with the 0.98
speaker_probe reports for 0.6B.
"""

from pathlib import Path

import numpy as np

MODEL_NAME = "Qwen/Qwen3-8B"  # 36 blocks, d_model 4096
DEVICE = "auto"  # "mps" | "cpu" | "cuda" | "auto"
# float32 is the reference numerics and what every earlier capture used, but 8B
# weights in fp32 are 32 GB before a single activation. "bfloat16" halves that
# and is the only way this fits a 32 GB node; pass --dtype on either
# model-loading script rather than editing this if you want to compare them.
DTYPE = "float32"
SEED = 0

# --- speaker boundary (mirrors speaker_probe) ---
N_CONVERSATIONS = 40
N_TURNS = 10
TOKENS_PER_TURN = 2
POOL = "shared"

# --- steering vector ---
# One trait, elicited by a system prompt over held-out questions from the eval
# itself. Difference in means between the trait prompt and a neutral one.
TRAIT = "psychopathy"  # key into data.TRAIT_SYSTEM
N_VECTOR = 80  # held-out eval items, each contributing a trait and a neutral prompt

# --- behavioural eval ---
# anthropics/evals persona set. Yes/No MCQ with a labelled matching answer; we
# score the *trait-consistent* choice, i.e. answer_not_matching_behavior for an
# agreeableness eval steered towards hostility.
EVAL_SET = "agreeableness"
# 60, not 150: accuracy over n items has a standard error of ~0.5/sqrt(n), so
# 60 resolves the swing this experiment is looking for (0 -> ~0.8) to +-6 points
# while costing 2.5x less than 150 in every cell of the sweep.
N_EVAL = 60
# Rows [0:N_EVAL] are scored; the vector is elicited on rows
# [N_EVAL:N_EVAL+N_VECTOR] instead, so the two never overlap.
VECTOR_OFFSET = N_EVAL

# --- sweep ---
# Layer index k refers to hidden_states[k]: k=0 is the embedding output, k>=1 is
# the residual stream leaving block k-1. Steering writes to the hook that produces
# exactly that tensor, so probe, vector and intervention all live at the same k.
# "all": add alpha*v at every token position (the usual steering setup, and the
#   only mode where "what fraction of activations crossed z=0" is a meaningful
#   quantity, since there are many activations to count).
# "last": add it only at the position the answer is read from, so the single
#   crossing coefficient alpha* stays exact and can be marked as a vertical line.
STEER_POSITIONS = "all"

# Three layers, not six, and none near either end of the 36 blocks. The sweep
# costs layers x coeffs x N_EVAL forward passes and the model is now 13x bigger,
# so this is where the budget actually goes. Nothing here needs a dense scan of
# depth: the early layers have no persona to move (a trait direction read off
# the embedding is token identity) and the last few write straight into the
# unembedding, so the question "does behaviour turn over near the speaker
# boundary" is asked in the middle third, where both the probe and CAA-style
# steering are known to work. boundary_geometry.png still shows a*, cos(w, v)
# and probe accuracy at *every* layer -- those come from the capture, not the
# sweep, and cost nothing. Widen this only once a middle layer has shown an
# effect worth localising.
STEER_LAYERS = (12, 18, 24)

# Same +-6 span as before, 15 points instead of 25, spent where the curve can
# bend. |v| is a consistent ~15% of the residual-stream norm at the swept
# layers, so step 0.5 is kept inside +-1.5 -- the window raw difference-in-means
# vectors are usually driven over (Tan et al. 2024; CAA uses +-1) -- and the
# tail beyond |2| is sampled coarsely, since by then the vector is a large
# fraction of the activation and the curve is saturated or destroyed, not
# structured. Must still span the median a* that analyze.py prints, or the
# boundary marker lands off-grid.
_HALF = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0])
COEFFS = np.concatenate([-_HALF[::-1], [0.0], _HALF])

HERE = Path(__file__).parent
CAPTURE_PATH = HERE / "outputs" / "capture.npz"
VECTOR_PATH = HERE / "outputs" / "vectors.npz"
STEER_PATH = HERE / "outputs" / "steer.npz"
DATA_DIR = HERE / "outputs" / "data"
PLOT_DIR = HERE / "outputs" / "plots"
