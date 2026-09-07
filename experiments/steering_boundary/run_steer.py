"""vectors.npz -> steer.npz: layer x coefficient sweep of the behavioural effect.

Loads the model (analysis never does). For each (layer, coefficient) it adds
alpha*v to the residual stream at every position and reads which of Yes/No the
model prefers on each eval question.

Usage: uv run python -m experiments.steering_boundary.run_steer
"""

import argparse

import numpy as np
import torch

from core import capture, model as model_mod

from . import config
from .data import eval_questions
from .run_capture import ANSWER_INSTRUCTION, answer_token_ids, prompt_ids


def hook_name(k: int) -> str:
    """Hook whose output *is* hidden_states[k].

    k=0 is the embedding output (blocks.0.hook_resid_pre); k>=1 is the residual
    stream leaving block k-1. Writing here means the probe, the steering vector
    and the intervention are all indexed by the same k — the whole point of
    alpha* being exact rather than approximate.
    """
    return "blocks.0.hook_resid_pre" if k == 0 else f"blocks.{k - 1}.hook_resid_post"


def add_vector(v: torch.Tensor):
    def fn(act, hook):
        return act + v
    return fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.MODEL_NAME)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--vectors", default=str(config.VECTOR_PATH))
    p.add_argument("--out", default=str(config.STEER_PATH))
    p.add_argument("--layers", type=int, nargs="*", default=list(config.STEER_LAYERS))
    p.add_argument(
        "--coeffs", type=float, nargs="*", default=None,
        help="steering coefficients; default config.COEFFS. Must span the median "
             "alpha* that analyze.py prints, or the boundary marker lands off-grid.",
    )
    args = p.parse_args()

    vec, meta = capture.load(args.vectors)
    V = vec["V"]
    coeffs = np.asarray(args.coeffs if args.coeffs else config.COEFFS, dtype=np.float64)

    m, tokenizer, device = model_mod.load(args.model, args.device, dtype=torch.float32)
    yes_id, no_id = answer_token_ids(tokenizer)
    assert (yes_id, no_id) == (meta["yes_id"], meta["no_id"]), "tokenizer drift since capture"

    rows = eval_questions(meta["eval_set"], meta["n_eval"], config.DATA_DIR)
    prompts = [prompt_ids(tokenizer, r["question"] + ANSWER_INSTRUCTION)[1] for r in rows]
    target_yes = np.array(
        [r["answer_not_matching_behavior"].strip() == "Yes" for r in rows], dtype=bool
    )

    acc = np.zeros((len(args.layers), len(coeffs)))
    for i, k in enumerate(args.layers):
        v = torch.tensor(V[k], dtype=torch.float32, device=device)
        for j, c in enumerate(coeffs):
            hooks = [] if c == 0 else [(hook_name(k), add_vector(c * v))]
            chose_yes = np.empty(len(prompts), dtype=bool)
            for n, ids in enumerate(prompts):
                logits = capture.run(
                    m, ids, device, what=("logits",), interventions=hooks
                )["logits"]
                chose_yes[n] = logits[-1, yes_id] > logits[-1, no_id]
            acc[i, j] = float((chose_yes == target_yes).mean())
        print(f"layer {k}: " + " ".join(f"{a:.2f}" for a in acc[i]))

    capture.save(
        args.out,
        {"layers": np.array(args.layers), "coeffs": coeffs, "acc": acc},
        {**meta, "n_prompts": len(prompts), "steer_device": device},
        compress=False,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
