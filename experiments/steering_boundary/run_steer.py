"""vectors.npz -> steer.npz: layer x coefficient sweep of the behavioural effect.

Loads the model (analysis never does). For each (layer, coefficient) it adds
alpha*v to the residual stream at every position and reads which of Yes/No the
model prefers on each eval question.

The sweep is layers x coeffs x prompts forward passes and the model is 8B, so
three things keep it affordable, none of which changes a number that comes out:
  - the c = 0 column is one unsteered pass per prompt, shared by every layer
    (no hook is registered there, so the forward cannot depend on k);
  - each pass captures only the residual stream of the layer being measured,
    not all 37;
  - "last"-position steering also drops every position but the read one.
config.STEER_LAYERS / COEFFS / N_EVAL carry the rest.

Usage: uv run python -m experiments.steering_boundary.run_steer
"""

import argparse

import numpy as np
import torch

from core import capture, model as model_mod

from . import config
from .data import eval_questions
from .run_capture import ANSWER_INSTRUCTION, answer_token_ids, prompt_ids

# k=0 is the embedding output, k>=1 the residual stream leaving block k-1.
# Writing at the hook that *is* hidden_states[k] means the probe, the steering
# vector and the intervention are all indexed by the same k -- the whole point
# of alpha* being exact rather than approximate.
hook_name = capture.resid_hook_name


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


def score(m, prompts, device, answer_ids, W, B, measure, hooks, steer_positions):
    """Run every eval prompt once and reduce to what a sweep cell records.

    `measure` is the list of layers k whose decision value z = w_k.h + b_k is
    tallied. A steered cell measures only the layer it steers; the unsteered
    pass measures all of them at once, which is free — the forward does not
    depend on k when no hook is registered.

    Returns (chose_yes, gap, crossed) where gap is the signed Yes-minus-No logit
    margin and crossed maps each k to the fraction of *steered* activations now
    on the assistant side of z = 0. Counting only what was pushed is what makes
    that ratio mean "of the things we moved, how many are past the boundary";
    measuring it rather than extrapolating z0 + c*(w.v) is algebraically the
    same at layer k but needs no assumption and survives a change to the hook.
    """
    yes_id, no_id = answer_ids
    # "last" reads and steers one position, so nothing else needs to come back.
    # "all" has to keep the position axis: the crossed fraction is counted over
    # every activation the hook pushed.
    pos = -1 if steer_positions == "last" else None

    chose_yes = np.empty(len(prompts), dtype=bool)
    gap = np.empty(len(prompts), dtype=np.float64)
    past = np.zeros(len(measure), dtype=np.int64)
    total = np.zeros(len(measure), dtype=np.int64)

    for n, ids in enumerate(prompts):
        out = capture.run(
            m, ids, device, what=("logits", "hidden_states"),
            interventions=hooks, layers=measure, positions=pos,
        )
        last = out["logits"][-1]
        chose_yes[n] = last[yes_id] > last[no_id]
        gap[n] = float(last[yes_id] - last[no_id])
        for i, k in enumerate(measure):
            z = out["hidden_states"][i].astype(np.float64) @ W[k] + B[k]
            past[i] += int((z > 0).sum())
            total[i] += z.size
    return chose_yes, gap, dict(zip(measure, past / total))


def report_baseline(chose_yes, target_yes):
    """The unsteered cell is also the sanity check on the whole experiment.

    A steering sweep can only move behaviour the model has. On Qwen3-0.6B this
    eval sat at 0.50 for every coefficient including 0, which is not "steering
    did nothing": it is the model answering the same word to nearly every
    question, against a Yes/No key that happens to be balanced. Print the
    Yes-rate next to the accuracy so the two are never confused again.
    """
    acc = float((chose_yes == target_yes).mean())
    yes_rate = float(chose_yes.mean())
    print(f"baseline (c=0): target-choice acc {acc:.3f}, "
          f"answered Yes on {yes_rate:.0%} of prompts "
          f"(key is {target_yes.mean():.0%} Yes)")
    if min(yes_rate, 1 - yes_rate) < 0.1:
        print("  WARNING: the unsteered model gives essentially one constant answer, so "
              "it is not doing the task and an accuracy near the key's Yes-rate is an "
              "artefact of that, not a behavioural baseline. A sweep on top of this "
              "measures nothing -- use a larger model before reading the curves.")
    return acc, yes_rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.MODEL_NAME)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--dtype", default=config.DTYPE, choices=("float32", "bfloat16", "float16"))
    p.add_argument("--vectors", default=str(config.VECTOR_PATH))
    p.add_argument("--out", default=str(config.STEER_PATH))
    p.add_argument("--layers", type=int, nargs="*", default=list(config.STEER_LAYERS))
    p.add_argument("--steer-positions", choices=("all", "last"),
                   default=config.STEER_POSITIONS)
    p.add_argument(
        "--n-eval", type=int, default=config.N_EVAL,
        help="eval prompts scored per cell; capped at what the capture holds. "
             "Every cell pays this, so it is the cheapest knob in the sweep.",
    )
    p.add_argument(
        "--coeffs", type=float, nargs="*", default=None,
        help="steering coefficients; default config.COEFFS. Must span the median "
             "alpha* that analyze.py prints, or the boundary marker lands off-grid.",
    )
    args = p.parse_args()

    vec, meta = capture.load(args.vectors)
    V, W, B = vec["V"], vec["W"], vec["B"]
    coeffs = np.asarray(args.coeffs if args.coeffs else config.COEFFS, dtype=np.float64)
    layers = list(args.layers)
    n_eval = min(args.n_eval, meta["n_eval"])

    m, tokenizer, device = model_mod.load(
        args.model, args.device, dtype=getattr(torch, args.dtype)
    )
    yes_id, no_id = answer_token_ids(tokenizer)
    assert (yes_id, no_id) == (meta["yes_id"], meta["no_id"]), "tokenizer drift since capture"

    rows = eval_questions(meta["eval_set"], n_eval, config.DATA_DIR)
    prompts = [prompt_ids(tokenizer, r["question"] + ANSWER_INSTRUCTION)[1] for r in rows]
    target_yes = np.array(
        [r["answer_not_matching_behavior"].strip() == "Yes" for r in rows], dtype=bool
    )
    n_steered = len(layers) * int((coeffs != 0).sum()) * len(prompts)
    print(f"{len(layers)} layers x {len(coeffs)} coeffs x {len(prompts)} prompts "
          f"= {n_steered + len(prompts)} forward passes "
          f"(was {len(layers) * len(coeffs) * len(prompts)} with c=0 re-run per layer)")

    acc = np.zeros((len(layers), len(coeffs)))
    crossed = np.zeros((len(layers), len(coeffs)))
    # Insurance against the metric, not a second result. acc is a hard argmax
    # between two fixed tokens and throws the magnitude away, so a real but
    # sub-threshold push reads as a flat curve. The signed margin toward the
    # target choice is the same measurement without the thresholding, and
    # recording it here costs one subtraction per prompt -- far cheaper than
    # discovering afterwards that the whole sweep has to be run again.
    margin = np.zeros((len(layers), len(coeffs)))

    def cell(hooks, measure):
        chose_yes, gap, cr = score(
            m, prompts, device, (yes_id, no_id), W, B, measure, hooks, args.steer_positions
        )
        signed = np.where(target_yes, gap, -gap)
        return chose_yes, float((chose_yes == target_yes).mean()), float(signed.mean()), cr

    # c = 0 registers no hook, so its forward pass is identical for every layer
    # in the sweep and running it once per layer was pure duplication. One pass
    # per prompt fills the whole column, measuring every layer's z as it goes.
    base_acc = base_yes = None
    zero = np.flatnonzero(coeffs == 0)
    if zero.size:
        j = int(zero[0])
        chose_yes, a, mg, cr = cell([], layers)
        base_acc, base_yes = report_baseline(chose_yes, target_yes)
        acc[:, j], margin[:, j] = a, mg
        crossed[:, j] = [cr[k] for k in layers]

    # The vector must carry the model's own dtype: adding a float32 v to a
    # bfloat16 activation promotes the residual stream to float32 and the next
    # block's matmul then fails against bfloat16 weights.
    dtype = next(m.parameters()).dtype
    for i, k in enumerate(layers):
        v = torch.tensor(V[k], dtype=dtype, device=device)
        for j, c in enumerate(coeffs):
            if c == 0:
                continue
            hooks = [(hook_name(k), add_vector(c * v, args.steer_positions))]
            _, acc[i, j], margin[i, j], cr = cell(hooks, [k])
            crossed[i, j] = cr[k]
        print(f"layer {k} acc:     " + " ".join(f"{a:.2f}" for a in acc[i]))
        print(f"layer {k} crossed: " + " ".join(f"{a:.2f}" for a in crossed[i]))
        print(f"layer {k} margin:  " + " ".join(f"{a:+.2f}" for a in margin[i]))

    capture.save(
        args.out,
        {
            "layers": np.array(layers), "coeffs": coeffs,
            "acc": acc, "crossed": crossed, "margin": margin,
        },
        {
            **meta, "steer_positions": args.steer_positions,
            "n_prompts": len(prompts), "steer_device": device,
            "steer_dtype": args.dtype,
            # The two numbers that say whether the sweep had anything to move.
            "baseline_acc": base_acc, "baseline_yes_rate": base_yes,
        },
        compress=False,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
