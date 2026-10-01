"""Config for the mirror-attribution experiment. Self-contained by design.

Question: if the residual stream is reflected across the self vector's boundary,
so that user tokens read as assistant tokens and vice versa, does the model then
misattribute who said what in the conversation?

The self vector is v_k = mu_asst - mu_user at hidden_states[k], with the
boundary at the midpoint of the two means: p = sigma(v.x + b),
b = -v.(mu_asst + mu_user)/2. It is fitted on speaker_probe's shared-pool
conversations, so N_TURNS / SEED / POOL below match speaker_probe's. The
behavioural eval is a separate dataset (opinions.py): n turns of stated
opinions, then "who said <statement>, the user or the assistant?".

MODEL_NAME matches speaker_probe's now that both default to Llama-3.1-8B-
Instruct. On Qwen it deliberately did not (0.6B there, 8B here), for the same
reason as steering_boundary: a mirror can only move behaviour the unmirrored
model has, and multi-turn attribution is the kind of task a 0.6B model answers
near chance. run_mirror prints the unmirrored accuracy first for that reason.
"""

from pathlib import Path

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # 32 blocks, 32 heads, d_model 4096
DEVICE = "auto"  # "mps" | "cpu" | "cuda" | "auto"
# float32 is the reference numerics; "bfloat16" halves the 32 GB of 8B weights.
# Pass --dtype on either model-loading script rather than editing this.
DTYPE = "float32"
SEED = 0

# --- self vector (mirrors speaker_probe) ---
N_CONVERSATIONS = 40
N_TURNS = 10
TOKENS_PER_TURN = 2
POOL = "shared"
TEST_SIZE = 0.25  # held-out conversations for the p = sigma(v.x + b) check

# --- attribution eval (opinions.py) ---
# n = conversation turns before the question; one (user, assistant) pair per
# turn, each on its own topic. Swept, so the plots show whether attribution and
# the mirror's effect depend on how much history there is.
EVAL_TURNS = (1, 2, 4, 8)
# Items per n. Accuracy over 60 items has a standard error of ~6 points, enough
# to resolve a flip from ~1 to ~0. Labels alternate, so half ask about a user
# statement and half about an assistant one.
N_EVAL = 60

# --- mirror sweep ---
# Layer index k refers to hidden_states[k] (see core.capture.resid_hook_name).
# One layer per pass: the reflection is applied once at k, and everything
# downstream reads the reflected stream. Middle-third layers plus a couple of
# early ones, because speaker_probe found the signal from layer 1 onward.
MIRROR_LAYERS = (2, 6, 12, 18, 24)
# Which positions the reflection touches. Position 0 (the attention sink) is
# never touched; the question and generation prompt are never touched.
#   "history":   every token of the n turns, both roles. A full flip predicts
#                accuracy -> 1 - baseline.
#   "assistant": only assistant blocks. Both speakers then look like the user.
#   "user":      only user blocks. Both speakers then look like the assistant.
MIRROR_MODES = ("history", "assistant", "user")

# Each self-vector cell is paired with a push along a random unit direction,
# of the same signed size per token as the reflection (2(v.x+b)/|v|). A large
# write in *any* direction can break the answer, so an effect only belongs to
# v if it beats this control.
RANDOM_CONTROL = True

# Cells where the median P(User) + P(Assistant) (either case) at the read
# position falls below this have stopped answering the question; the plots grey
# them out.
MIN_ANSWER_MASS = 0.5

HERE = Path(__file__).parent
CAPTURE_PATH = HERE / "outputs" / "capture.npz"
VECTOR_PATH = HERE / "outputs" / "vectors.npz"
MIRROR_PATH = HERE / "outputs" / "mirror.npz"
PLOT_DIR = HERE / "outputs" / "plots"
