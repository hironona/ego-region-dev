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


def add_vector(v: torch.Tensor, positions: str):
    """Write hook adding v to the residual stream.

    "all" broadcasts over the position axis — the ordinary steering setup, where
    every token in the prompt is pushed. "last" touches only the position the
    answer is read from, which keeps the single crossing coefficient alpha*
    exact for that one activation.
    """
    assert positions in ("all", "last"), positions

    def fn(act, hook):
        if positions == "all":
            return act + v
        out = act.clone()
        out[:, -1, :] = out[:, -1, :] + v
        return out

    return fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.MODEL_NAME)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--vectors", default=str(config.VECTOR_PATH))
    p.add_argument("--out", default=str(config.STEER_PATH))
    p.add_argument("--layers", type=int, nargs="*", default=list(config.STEER_LAYERS))
    p.add_argument("--steer-positions", choices=("all", "last"),
                   default=config.STEER_POSITIONS)
    p.add_argument(
        "--coeffs", type=float, nargs="*", default=None,
        help="steering coefficients; default config.COEFFS. Must span the median "
             "alpha* that analyze.py prints, or the boundary marker lands off-grid.",
    )
    args = p.parse_args()

    vec, meta = capture.load(args.vectors)
    V, W, B = vec["V"], vec["W"], vec["B"]
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
    crossed = np.zeros((len(args.layers), len(coeffs)))
    for i, k in enumerate(args.layers):
        v = torch.tensor(V[k], dtype=torch.float32, device=device)
        w_k, b_k = W[k], B[k]
        for j, c in enumerate(coeffs):
            hooks = (
                [] if c == 0
                else [(hook_name(k), add_vector(c * v, args.steer_positions))]
            )
            chose_yes = np.empty(len(prompts), dtype=bool)
            n_past, n_total = 0, 0
            for n, ids in enumerate(prompts):
                out = capture.run(
                    m, ids, device,
                    what=("logits", "hidden_states"), interventions=hooks,
                )
                logits = out["logits"]
                chose_yes[n] = logits[-1, yes_id] > logits[-1, no_id]
                # Count only the activations that were actually steered, so the
                # ratio means "of the things we pushed, how many are now on the
                # assistant side". Measured rather than extrapolated from
                # z0 + c*(w.v) — algebraically identical at layer k, but it needs
                # no assumption and stays correct if the hook ever changes.
                h = out["hidden_states"][k].astype(np.float64)
                if args.steer_positions == "last":
                    h = h[-1:]
                z = h @ w_k + b_k
                n_past += int((z > 0).sum())
                n_total += z.size
            acc[i, j] = float((chose_yes == target_yes).mean())
            crossed[i, j] = n_past / n_total
        print(f"layer {k} acc:     " + " ".join(f"{a:.2f}" for a in acc[i]))
        print(f"layer {k} crossed: " + " ".join(f"{a:.2f}" for a in crossed[i]))

    capture.save(
        args.out,
        {
            "layers": np.array(args.layers), "coeffs": coeffs,
            "acc": acc, "crossed": crossed,
        },
        {
            **meta, "steer_positions": args.steer_positions,
            "n_prompts": len(prompts), "steer_device": device,
        },
        compress=False,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
