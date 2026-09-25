"""vectors.npz -> steer.npz: layer x scale sweep of the behavioural effect.

Loads the model (analysis never does). For each (layer, scale) it adds c*v to
the residual stream at every position, with c = scale * |h_k| / |v_k|, and reads
which of Yes/No the model prefers on each eval question. The same c is also
applied along a random direction of the same norm (config.RANDOM_CONTROL), since
an effect only belongs to the trait vector if it beats that control.

Each cell records more than accuracy, because on a 50%-Yes key accuracy alone
reads three different things as the same 0.5: a real half-way effect, one
constant answer, and a model that no longer answers at all. So it also stores
the accuracy on each half of the key, the Yes-rate, and the median answer mass
P(Yes) + P(No).

The sweep is layers x coeffs x prompts forward passes and the model is 8B, so
three things keep it affordable, none of which changes a number that comes out:
  - the c = 0 column is one unsteered pass per prompt, shared by every layer
    and by both vector kinds (no hook is registered there, so the forward
    cannot depend on k);
  - each pass captures only the residual stream of the layer being measured,
    not all 37;
  - "last"-position steering also drops every position but the read one.
config.STEER_LAYERS / STEER_SCALES / N_EVAL carry the rest.

Usage: uv run python -m experiments.steering_boundary.run_steer
"""

import argparse

import numpy as np
import torch

from core import capture, model as model_mod

from . import config
from .data import eval_questions
from .run_capture import ANSWER_INSTRUCTION, PROMPT_STYLE, answer_token_ids, prompt_ids

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


def random_direction(k, like, seed):
    """Random unit direction scaled to |like|, fixed by (seed, k).

    Seeded per layer, so the control at layer k does not change when --layers
    adds or drops another layer.
    """
    r = np.random.default_rng([seed, k]).standard_normal(like.shape)
    return r * (np.linalg.norm(like) / np.linalg.norm(r))


def answer_mass(logits, answer_ids):
    """P(Yes) + P(No) under the full softmax of one logit row."""
    x = logits.astype(np.float64)
    lse = x.max() + np.log(np.exp(x - x.max()).sum())
    return float(np.exp(x[list(answer_ids)] - lse).sum())


def score(m, prompts, device, answer_ids, W, B, measure, hooks, steer_positions):
    """Run every eval prompt once and reduce to what a sweep cell records.

    `measure` is the list of layers k whose decision value z = w_k.h + b_k is
    tallied. A steered cell measures only the layer it steers; the unsteered
    pass measures all of them at once, which is free — the forward does not
    depend on k when no hook is registered.

    Returns (chose_yes, gap, mass, crossed). gap is the signed Yes-minus-No logit
    margin, mass is P(Yes)+P(No) per prompt, and crossed maps each k to the
    fraction of *steered* activations now on the assistant side of z = 0.
    Counting only what was pushed is what makes that ratio mean "of the things we
    moved, how many are past the boundary". Measuring it, rather than
    extrapolating z0 + c*(w.v), gives the same number at layer k but needs no
    assumption and keeps working if the hook changes.
    """
    yes_id, no_id = answer_ids
    # "last" reads and steers one position, so nothing else needs to come back.
    # "all" has to keep the position axis: the crossed fraction is counted over
    # every activation the hook pushed.
    pos = -1 if steer_positions == "last" else None

    chose_yes = np.empty(len(prompts), dtype=bool)
    gap = np.empty(len(prompts), dtype=np.float64)
    mass = np.empty(len(prompts), dtype=np.float64)
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
        mass[n] = answer_mass(last, answer_ids)
        for i, k in enumerate(measure):
            z = out["hidden_states"][i].astype(np.float64) @ W[k] + B[k]
            past[i] += int((z > 0).sum())
            total[i] += z.size
    return chose_yes, gap, mass, dict(zip(measure, past / total))


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


METRICS = ("acc", "acc_tgt_yes", "acc_tgt_no", "yes_rate", "answer_mass", "margin", "crossed")


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
        "--scales", type=float, nargs="*", default=None,
        help="steering scales |c v| / |h| (default config.STEER_SCALES). The raw "
             "coefficient at layer k is scale * h_norm[k] / |v_k|.",
    )
    p.add_argument(
        "--no-control", action="store_true",
        help="skip the random-direction control (halves the steered passes)",
    )
    args = p.parse_args()

    vec, meta = capture.load(args.vectors)
    if "h_norm" not in vec:
        raise SystemExit(f"{args.vectors} has no h_norm; re-run analyze first.")
    V, W, B, h_norm = vec["V"], vec["W"], vec["B"], vec["h_norm"]
    # W, B and V were all fitted on the capture's prompts, and this sweep steers
    # the prompt built below. If the two prompts differ, each of them is anchored
    # to a different activation, and the sweep would still produce a
    # plausible-looking plot. This is checked before the model loads: at 8B
    # loading takes minutes, and the check only needs the sidecar json.
    assert meta.get("prompt_style") == PROMPT_STYLE, (
        f"capture used prompt_style {meta.get('prompt_style')!r}, this build expects "
        f"{PROMPT_STYLE!r}. Re-run run_capture (FORCE_CAPTURE=1 in the slurm job) "
        f"and analyze before sweeping."
    )
    scales = np.asarray(args.scales if args.scales else config.STEER_SCALES, dtype=np.float64)
    layers = list(args.layers)
    kinds = ["trait"] if args.no_control or not config.RANDOM_CONTROL else ["trait", "random"]
    n_eval = min(args.n_eval, meta["n_eval"])
    # The raw coefficient per (layer, scale). Both kinds share it: the random
    # direction has |v_k| by construction.
    coeffs = np.array(
        [scales * h_norm[k] / np.linalg.norm(V[k]) for k in layers], dtype=np.float64
    )

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
    n_steered = len(kinds) * len(layers) * int((scales != 0).sum()) * len(prompts)
    print(f"{len(kinds)} kinds ({', '.join(kinds)}) x {len(layers)} layers x "
          f"{len(scales)} scales x {len(prompts)} prompts = "
          f"{n_steered + len(prompts)} forward passes (c=0 shared)")

    res = {name: np.zeros((len(kinds), len(layers), len(scales))) for name in METRICS}

    def cell(hooks, measure):
        chose_yes, gap, mass, cr = score(
            m, prompts, device, (yes_id, no_id), W, B, measure, hooks, args.steer_positions
        )
        hit = chose_yes == target_yes
        # The margin is kept as a check on the metric, not as a second result.
        # acc is a hard argmax between two tokens and throws the magnitude away,
        # so a real push that stays below the flip threshold reads as a flat
        # curve. The signed margin toward the target is the same measurement
        # without the threshold.
        signed = np.where(target_yes, gap, -gap)
        return chose_yes, {
            "acc": float(hit.mean()),
            "acc_tgt_yes": float(hit[target_yes].mean()),
            "acc_tgt_no": float(hit[~target_yes].mean()),
            "yes_rate": float(chose_yes.mean()),
            "answer_mass": float(np.median(mass)),
            "margin": float(signed.mean()),
        }, cr

    def put(kind_i, layer_i, scale_j, metrics, crossed):
        for name, value in metrics.items():
            res[name][kind_i, layer_i, scale_j] = value
        res["crossed"][kind_i, layer_i, scale_j] = crossed

    # c = 0 registers no hook, so its forward pass is identical for every layer
    # and both kinds. One pass per prompt fills that whole column, measuring
    # every layer's z as it goes.
    base_acc = base_yes = None
    zero = np.flatnonzero(scales == 0)
    if zero.size:
        j = int(zero[0])
        chose_yes, metrics, cr = cell([], layers)
        base_acc, base_yes = report_baseline(chose_yes, target_yes)
        for t in range(len(kinds)):
            for i, k in enumerate(layers):
                put(t, i, j, metrics, cr[k])

    # The vector must have the model's own dtype. Adding a float32 v to a
    # bfloat16 activation promotes the residual stream to float32, and the next
    # block's matmul then fails against the bfloat16 weights.
    dtype = next(m.parameters()).dtype
    for i, k in enumerate(layers):
        directions = {"trait": V[k], "random": random_direction(k, V[k], config.SEED)}
        for t, kind in enumerate(kinds):
            v = torch.tensor(directions[kind], dtype=dtype, device=device)
            for j, c in enumerate(coeffs[i]):
                if scales[j] == 0:
                    continue
                hooks = [(hook_name(k), add_vector(float(c) * v, args.steer_positions))]
                _, metrics, cr = cell(hooks, [k])
                put(t, i, j, metrics, cr[k])
            print(f"layer {k} {kind:6s} acc|tgt=Yes  " + " ".join(f"{a:5.2f}" for a in res["acc_tgt_yes"][t, i]))
            print(f"layer {k} {kind:6s} acc|tgt=No   " + " ".join(f"{a:5.2f}" for a in res["acc_tgt_no"][t, i]))
            print(f"layer {k} {kind:6s} yes-rate     " + " ".join(f"{a:5.2f}" for a in res["yes_rate"][t, i]))
            print(f"layer {k} {kind:6s} P(Yes)+P(No) " + " ".join(f"{a:5.2f}" for a in res["answer_mass"][t, i]))
        print(f"layer {k} scales              " + " ".join(f"{a:+.2f}" for a in scales))

    capture.save(
        args.out,
        {"layers": np.array(layers), "scales": scales, "coeffs": coeffs, **res},
        {
            **meta, "steer_positions": args.steer_positions,
            "n_prompts": len(prompts), "steer_device": device,
            "steer_dtype": args.dtype,
            "kinds": kinds, "random_seed": config.SEED,
            "min_answer_mass": config.MIN_ANSWER_MASS,
            # The two numbers that say whether the sweep had anything to move.
            "baseline_acc": base_acc, "baseline_yes_rate": base_yes,
        },
        compress=False,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
