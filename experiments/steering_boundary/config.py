"""Config for the steering-vs-speaker-boundary experiment. Self-contained by design.

Question: is the speaker probe's decision surface (z = w.h + b = 0, the learned
user/assistant boundary) related to where a trait steering vector starts changing
behaviour? Steering by alpha*v moves z by alpha*(w.v), so each eval prompt has an
exact crossing coefficient alpha* = -z(h) / (w.v). We plot the steering effect
against alpha and mark alpha* on it.

N_TURNS / SEED / POOL match speaker_probe's, so the speaker boundary fitted
here is built the same way as the one measured there. MODEL_NAME now matches
too (both default to Llama-3.1-8B-Instruct). On Qwen it deliberately did not:
speaker_probe ran on Qwen3-0.6B, and at that size the unsteered model answers
the agreeableness eval at chance (it picks one of Yes/No largely regardless of
the question), so a steering sweep on top of it has no behaviour to move; this
experiment ran on Qwen3-8B instead. run_steer prints the unsteered Yes-rate for
exactly this reason. The probe is refitted on this experiment's own capture
(40 conversations, 2 tokens per turn), so its accuracy is still not the same
number as speaker_probe's.
"""

from pathlib import Path

import numpy as np

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # 32 blocks, 32 heads, d_model 4096
# Earlier results (the numbers quoted below) are from "Qwen/Qwen3-8B" (36 blocks).
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
# "caa": contrastive activation addition on the eval's own answer tokens. Each
#   held-out question is followed by the trait-consistent answer (label 1) and by
#   the other answer (label 0), and the residual stream is read *at the answer
#   token*. Rows are chosen so "Yes" is the trait answer exactly as often as "No",
#   so the answer word cancels out of the difference in means and what is left is
#   "gave the callous answer".
# "system_prompt": the earlier method, kept so the old result stays reproducible.
#   Trait vs neutral system prompt, read at the last prompt token. On Qwen3-8B the
#   prompt itself flips 98% of answers, but the vector it yields does not: its
#   effect on the eval was indistinguishable from a random direction of the same
#   norm. The flip is question-dependent (Yes on callous statements, No on kind
#   ones), so on a balanced key it cancels out of the mean; only tone survived.
VECTOR_METHOD = "caa"
TRAIT = "psychopathy"  # key into data.TRAIT_SYSTEM; used by "system_prompt" only
N_VECTOR = 80  # held-out eval items; each contributes one prompt per label

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
#   Not a working alternative on Qwen3-8B: with the system-prompt vector at layer 18 it
#   changed no answer at all up to 1.75x the activation norm (presumably the
#   later layers read the answer off the unsteered prompt tokens via attention).
STEER_POSITIONS = "all"

# Three layers, not six, and none near either end of the 32 blocks. The sweep
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

# The grid is in units of the residual stream, not of v: a scale r steers with
# c = r * |h_k| / |v_k|, where |h_k| is the median norm at the read position of
# the unsteered eval prompts (vectors.npz "h_norm"). A raw coefficient means
# something different at every layer and for every vector -- |v|/|h| was ~0.15
# on Qwen3-0.6B but 0.17 / 0.29 / 0.36 at layers 12 / 18 / 24 on Qwen3-8B, so the old +-12
# grid pushed 2-4x the whole activation and most of it measured a broken model.
# On Qwen3-8B the answer format survives up to r ~ 1 and is gone by r ~ 1.7; 1.5 is
# kept at the edge so the plot shows where that happens.
_HALF = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5])
STEER_SCALES = np.concatenate([-_HALF[::-1], [0.0], _HALF])

# Every trait cell is paired with the same coefficient along a random direction
# of the same norm (seeded by SEED). Steering along *any* large direction pushed
# Qwen3-8B towards a question-independent "Yes", so an effect only counts as the
# trait's if it beats this control. Doubles the steered passes.
RANDOM_CONTROL = True

# Cells where the median P(Yes) + P(No) at the read position falls below this
# are not answering the question any more. Their "accuracy" is an argmax between
# two tail tokens -- on a 50%-Yes key it comes out at exactly 0.5 -- so the plot
# greys them out instead of drawing them as a result.
MIN_ANSWER_MASS = 0.5

HERE = Path(__file__).parent
CAPTURE_PATH = HERE / "outputs" / "capture.npz"
VECTOR_PATH = HERE / "outputs" / "vectors.npz"
STEER_PATH = HERE / "outputs" / "steer.npz"
DATA_DIR = HERE / "outputs" / "data"
PLOT_DIR = HERE / "outputs" / "plots"
