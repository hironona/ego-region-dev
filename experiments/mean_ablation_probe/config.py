"""Config for the mean-ablation speaker-probe experiment.

Localises where the user/assistant signal is written into the residual stream by
mean-ablating a sliding window of attention or MLP blocks and re-probing.
"""

from pathlib import Path

from experiments.self_user_geometry.config import DEVICE, MODEL_NAME  # noqa: F401

# The dataset is the speaker-probe dataset, verbatim: same builder, same seed,
# same shared sentence pool. Only the sampling density differs (see below).
from experiments.speaker_probe.config import (  # noqa: F401
    N_CONVERSATIONS,
    N_TURNS,
    POOL,
    SEED,
)

# 13 conditions x this many rows x 29 layers x 1024 dims of fp16 is the whole disk
# cost of the experiment: 2 tokens/turn -> ~4k rows -> ~240 MB per condition.
# Raise it if a probe ever looks sample-starved; the effect here is not subtle.
TOKENS_PER_TURN = 2

WINDOW = 5  # consecutive blocks frozen per condition
COMPONENTS = ("attn", "mlp")

HERE = Path(__file__).parent
CAPTURE_DIR = HERE / "outputs" / "captures"
PLOT_DIR = HERE / "outputs" / "plots"
