"""vectors.npz -> mirror.npz: attribution accuracy with the speaker mirrored.

Loads the model (analysis never does). For every n in config.EVAL_TURNS it
builds the opinion dataset, scores the unmirrored model once, then for each
(layer k, mode, kind) re-runs every item with a write hook at hidden_states[k]:

  kind "self":   Householder reflection across the self vector's boundary,
                 x' = x - 2 (v.x + b) / |v|^2 * v. It negates z = v.x + b
                 exactly and leaves everything orthogonal to v alone, so an
                 assistant token lands where the matching user token would be.
  kind "random": the same signed push size per token, 2 (v.x + b) / |v|,
                 along a random unit direction instead (config.RANDOM_CONTROL).

The answer is read as a forced choice at the first answer position: log P of
"User" (the user said it) against log P of "Assistant" (the assistant did). Each
item stores both log-probs and the answer mass, so analyze can compute
accuracy, flip rate and margin without a second pass.

Usage: uv run python -m experiments.mirror_attribution.run_mirror --device cuda
"""

import argparse

import numpy as np
import torch

from core import capture, model as model_mod

from . import config
from .opinions import build

hook_name = capture.resid_hook_name

# Indexed by label: 0 = the user said it, 1 = the assistant did. Both cases
# count, since "user" and "User" are the same answer. Role names, not "You"/"Me":
# see opinions.QUESTION.
ANSWER_WORDS = (("User", "user"), ("Assistant", "assistant"))
GEN_PROMPT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build_prompt(tokenizer, item):
    """Templated history + question, ending where the answer begins.

    Returns (ids (1, T) tensor, role (T,) int8) with role 0/1 for every token of
    a user/assistant history block (header, content, <|im_end|>, newline) and
    -1 for the question and the generation prompt, which no mode touches.

    Two Qwen3 template facts are pinned here (see CLAUDE.md). History: the
    `<think>` block is only injected into assistant messages *after* the last
    user query, and the question is that query, so the history comes out clean.
    Generation prompt: `enable_thinking=False` is what appends the pre-closed
    `<think>\\n\\n</think>\\n\\n`; without it the model starts reasoning and the
    answer logits are read from the tail of the distribution.
    """
    msgs = item["messages"] + [{"role": "user", "content": item["question"]}]
    text = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    blocks = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs]
    assert text == "".join(blocks) + GEN_PROMPT, repr(text[-200:])
    assert text.count("<think>") == 1, "history picked up a <think> block"

    ids, role = [], []
    for msg, block in zip(msgs[:-1], blocks[:-1]):
        block_ids = tokenizer(block, add_special_tokens=False)["input_ids"]
        ids += block_ids
        role += [0 if msg["role"] == "user" else 1] * len(block_ids)
    tail = tokenizer(blocks[-1] + GEN_PROMPT, add_special_tokens=False)["input_ids"]
    ids += tail
    role += [-1] * len(tail)
    # Per-block tokenisation is only valid if it matches the full text's: every
    # block starts with <|im_start|>, so merges cannot cross a block boundary.
    assert ids == tokenizer(text, add_special_tokens=False)["input_ids"]
    return torch.tensor([ids]), np.array(role, dtype=np.int8)


def mode_positions(role, mode):
    """Token positions a mode reflects. Position 0 (the attention sink, whose
    norm is far larger than any content token's) is never touched."""
    keep = {"history": role >= 0, "user": role == 0, "assistant": role == 1}[mode]
    keep = keep.copy()
    keep[0] = False
    return np.flatnonzero(keep)


def answer_token_ids(tokenizer):
    """[[ids for label 0], [ids for label 1]]; every word must be one token."""
    out = []
    for words in ANSWER_WORDS:
        ids = [tokenizer(w, add_special_tokens=False)["input_ids"] for w in words]
        assert all(len(i) == 1 for i in ids), (words, ids)
        out.append([i[0] for i in ids])
    flat = [i for ids in out for i in ids]
    assert len(set(flat)) == len(flat), flat
    return out


def answer_logp(logits, answer_ids):
    """(log P(label 0 answer), log P(label 1 answer)), answer mass) for one row.

    Log-probs under the full softmax, summed over each label's words, so the
    mass over all answer words says whether the model is answering at all.
    """
    x = torch.as_tensor(logits, dtype=torch.float64).log_softmax(-1)
    lp = np.array([float(torch.logsumexp(x[ids], 0)) for ids in answer_ids])
    return lp, float(np.exp(lp).sum())


def mirror_hook(pos, v, b, push=None, stats=None):
    """Write hook reflecting positions `pos` across v.x + b = 0.

    `push=None` is the Householder reflection. A unit vector `push` gives the
    random control: the same signed step 2 (v.x + b) / |v| per token, along
    `push` instead of v, so both kinds move every token by exactly the same
    distance and differ only in direction.

    `stats`, if given, collects the pre-mirror z and |x' - x| / |x| of every
    touched token. Computed in float32 whatever the model's dtype, and written
    back in the model's dtype (a float32 residual stream would make the next
    bfloat16 block's matmul fail).
    """
    vv = float(v @ v)

    def fn(act, hook):
        idx = torch.as_tensor(pos, dtype=torch.long, device=act.device)
        x = act[:, idx, :].float()
        z = x @ v + b
        if push is None:
            delta = (-2.0 * z / vv)[..., None] * v
        else:
            delta = (-2.0 * z / vv ** 0.5)[..., None] * push
        out = act.clone()
        out[:, idx, :] = (x + delta).to(act.dtype)
        if stats is not None:
            stats["z"].append(z[0].cpu().numpy())
            stats["disp"].append((delta.norm(dim=-1) / x.norm(dim=-1))[0].cpu().numpy())
        return out

    return fn


def random_unit(k, d, seed):
    """Random unit direction, fixed by (seed, k) so --layers subsets agree."""
    r = np.random.default_rng([seed, k]).standard_normal(d)
    return r / np.linalg.norm(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.MODEL_NAME)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--dtype", default=config.DTYPE, choices=("float32", "bfloat16", "float16"))
    p.add_argument("--vectors", default=str(config.VECTOR_PATH))
    p.add_argument("--out", default=str(config.MIRROR_PATH))
    p.add_argument("--turns", type=int, nargs="*", default=list(config.EVAL_TURNS))
    p.add_argument("--n-eval", type=int, default=config.N_EVAL)
    p.add_argument("--layers", type=int, nargs="*", default=list(config.MIRROR_LAYERS))
    p.add_argument("--modes", nargs="*", default=list(config.MIRROR_MODES),
                   choices=("history", "assistant", "user"))
    p.add_argument("--no-control", action="store_true",
                   help="skip the random-direction control (halves the mirrored passes)")
    args = p.parse_args()

    vec, vmeta = capture.load(args.vectors)
    # Checked before the model loads: a self vector from another model's
    # residual stream would still produce a plausible plot.
    if vmeta["model"] != args.model:
        raise SystemExit(f"{args.vectors} was fitted on {vmeta['model']}, not {args.model}")
    V, B = vec["V"], vec["B"]
    kinds = ["self"] if args.no_control or not config.RANDOM_CONTROL else ["self", "random"]
    turns, layers, modes = list(args.turns), list(args.layers), list(args.modes)

    m, tokenizer, device = model_mod.load(
        args.model, args.device, dtype=getattr(torch, args.dtype)
    )
    answer_ids = answer_token_ids(tokenizer)
    d = m.cfg.d_model

    n_cells = len(layers) * len(modes) * len(kinds)
    print(f"{len(turns)} n x {args.n_eval} items x (1 + {len(layers)} layers x "
          f"{len(modes)} modes x {len(kinds)} kinds) = "
          f"{len(turns) * args.n_eval * (1 + n_cells)} forward passes")

    shape = (len(turns), len(kinds), len(layers), len(modes), args.n_eval)
    labels = np.zeros((len(turns), args.n_eval), dtype=np.uint8)
    base_logp = np.zeros((len(turns), args.n_eval, 2))
    base_mass = np.zeros((len(turns), args.n_eval))
    logp = np.zeros(shape + (2,))
    mass = np.zeros(shape)
    disp = np.zeros(shape[:1] + shape[2:4])  # median |x'-x|/|x|, identical for both kinds
    n_touched = np.zeros(shape[:1] + shape[2:4], dtype=np.int64)
    # Pre-mirror z sign vs role on the touched tokens of the "history" mode:
    # the self vector's accuracy in situ, headers and all, at the mirrored layer.
    z_acc = np.full((len(turns), len(layers), 2), np.nan)

    def run(ids, hooks):
        out = capture.run(m, ids, device, what=("logits",), interventions=hooks, positions=-1)
        return answer_logp(out["logits"][0], answer_ids)

    for a, n in enumerate(turns):
        items = build(args.n_eval, n, config.SEED)
        prompts = [build_prompt(tokenizer, it) for it in items]
        labels[a] = [it["label"] for it in items]

        for j, (ids, _role) in enumerate(prompts):
            base_logp[a, j], base_mass[a, j] = run(ids, [])
        base_hit = (base_logp[a, :, 1] > base_logp[a, :, 0]) == labels[a]
        print(f"n={n} unmirrored: acc {base_hit.mean():.3f}, "
              f"median answer mass {np.median(base_mass[a]):.3f}")
        if np.median(base_mass[a]) < config.MIN_ANSWER_MASS or base_hit.mean() < 0.7:
            print("  WARNING: the unmirrored model is not doing the task at this n, so a "
                  "mirrored accuracy here has nothing to flip.")

        for i, k in enumerate(layers):
            v = torch.tensor(V[k], dtype=torch.float32, device=device)
            directions = {
                "self": None,
                "random": torch.tensor(random_unit(k, d, config.SEED),
                                       dtype=torch.float32, device=device),
            }
            for c, mode in enumerate(modes):
                for t, kind in enumerate(kinds):
                    stats = {"z": [], "disp": []}
                    roles = []
                    for j, (ids, role) in enumerate(prompts):
                        pos = mode_positions(role, mode)
                        roles.append(role[pos])
                        hooks = [(hook_name(k), mirror_hook(pos, v, float(B[k]),
                                                            directions[kind], stats))]
                        logp[a, t, i, c, j], mass[a, t, i, c, j] = run(ids, hooks)
                    if kind == "self":
                        z = np.concatenate(stats["z"])
                        disp[a, i, c] = np.median(np.concatenate(stats["disp"]))
                        n_touched[a, i, c] = z.size
                        if mode == "history":
                            r = np.concatenate(roles)
                            z_acc[a, i] = [((z > 0) == r)[r == s].mean() for s in (0, 1)]
                    hit = (logp[a, t, i, c, :, 1] > logp[a, t, i, c, :, 0]) == labels[a]
                    print(f"n={n} k={k:2d} {mode:9s} {kind:6s} acc {hit.mean():.3f}  "
                          f"mass {np.median(mass[a, t, i, c]):.3f}  "
                          f"|dx|/|x| {disp[a, i, c]:.3f}")

    capture.save(
        args.out,
        {
            "turns": np.array(turns), "layers": np.array(layers),
            "labels": labels, "base_logp": base_logp, "base_mass": base_mass,
            "logp": logp, "mass": mass, "disp": disp, "n_touched": n_touched,
            "z_acc": z_acc,
        },
        {
            "model": args.model, "device": device, "dtype": args.dtype,
            "vectors": args.vectors, "n_eval": args.n_eval, "seed": config.SEED,
            "kinds": kinds, "modes": modes, "answer_words": ANSWER_WORDS,
            "min_answer_mass": config.MIN_ANSWER_MASS,
            "label_convention": (
                "labels: 0=user said it (correct answer User), 1=assistant (Assistant). "
                "logp[..., 0|1] = log P(label 0|1 answer). "
                "z_acc[..., 0|1] = fraction of user|assistant history tokens on "
                "their own side of v.x+b=0 before mirroring."
            ),
        },
        compress=False,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
