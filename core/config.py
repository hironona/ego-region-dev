"""Model and device defaults shared by every experiment.

Experiment-agnostic on purpose: which model to load and where to run it is not a
property of any one experiment, and every `experiments/<name>/config.py` imports
these rather than repeating them. Prompts, roles, datasets and analysis choices
stay in the experiment.
"""

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "auto"  # "mps" | "cpu" | "cuda" | "auto"
