"""Per-head mean ablation over the early attention layers -> one capture each.

Conditions: baseline, every single head in SWEEP_LAYERS, each whole layer, and
the whole window (the previous experiment's positive control).

Usage: uv run python -m experiments.head_ablation_sweep.run_capture
       uv run python -m experiments.head_ablation_sweep.run_capture --n 100
"""

import argparse

import torch

from core import capture, model as model_mod

# Same dataset and same capture loop as the window sweep, imported rather than
# re-implemented so the two experiments' numbers mean the same thing.
from experiments.mean_ablation_probe.run_capture import capture_condition
from experiments.speaker_probe.conversations import build

from . import config

Z_HOOK = "blocks.{layer}.attn.hook_z"


def mean_ablate_heads(heads):
    """Hook fn freezing `heads` of one layer: each keeps its context-mean output.

    hook_z is (batch, pos, n_heads, d_head) — the per-head output before W_O.
    Because W_O is linear, freezing every head of a layer this way is exactly
    equivalent to freezing that layer's attn_out, which is what the window sweep
    did; freezing a subset is the finer-grained version of the same edit.
    """

    def fn(act, hook):
        act = act.clone()
        idx = list(heads)
        act[:, :, idx, :] = act[:, :, idx, :].mean(dim=1, keepdim=True)
        return act

    return fn


def conditions(sweep_layers, n_heads):
    """[(name, interventions)] — baseline, single heads, whole layers, whole window."""
    out = [("baseline", [])]
    for layer in sweep_layers:
        for head in range(n_heads):
            out.append(
                (f"L{layer:02d}H{head:02d}", [(Z_HOOK.format(layer=layer), mean_ablate_heads([head]))])
            )
    for layer in sweep_layers:
        out.append(
            (f"L{layer:02d}all", [(Z_HOOK.format(layer=layer), mean_ablate_heads(range(n_heads)))])
        )
    window = [
        (Z_HOOK.format(layer=layer), mean_ablate_heads(range(n_heads))) for layer in sweep_layers
    ]
    out.append((f"window_{min(sweep_layers):02d}-{max(sweep_layers):02d}", window))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=config.N_CONVERSATIONS)
    p.add_argument("--turns", type=int, default=config.N_TURNS)
    p.add_argument("--tokens-per-turn", type=int, default=config.TOKENS_PER_TURN)
    p.add_argument("--sweep-layers", type=int, nargs="+", default=list(config.SWEEP_LAYERS))
    p.add_argument("--probe-layers", type=int, nargs="+", default=list(config.PROBE_LAYERS))
    p.add_argument("--model", default=config.MODEL_NAME)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--out-dir", default=str(config.CAPTURE_DIR))
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--pool", choices=("shared", "role"), default=config.POOL)
    args = p.parse_args()

    m, tokenizer, device = model_mod.load(args.model, args.device, dtype=torch.float32)
    conversations = build(args.n, args.turns, args.seed, pool=args.pool)
    todo = conditions(args.sweep_layers, m.cfg.n_heads)
    print(
        f"device={device}  n_heads={m.cfg.n_heads}  conditions={len(todo)}  "
        f"probe_layers={args.probe_layers}"
    )

    for k, (name, interventions) in enumerate(todo, 1):
        arrays = capture_condition(
            m,
            tokenizer,
            device,
            conversations,
            interventions,
            args.tokens_per_turn,
            layers=args.probe_layers,
        )
        meta = {
            "model": args.model,
            "device": device,
            "condition": name,
            "ablated_hooks": sorted({h for h, _ in interventions}),
            "ablation": "mean over context positions, per head (hook_z)",
            "probe_layers": list(args.probe_layers),
            "n_conversations": args.n,
            "n_turns": args.turns,
            "tokens_per_turn": args.tokens_per_turn,
            "n_layers": m.cfg.n_layers,
            "n_heads": m.cfg.n_heads,
            "d_model": m.cfg.d_model,
            "pool": args.pool,
            "seed": args.seed,
            "label_convention": "speaker: 0=user, 1=assistant",
        }
        path = f"{args.out_dir}/{name}.npz"
        capture.save(path, arrays, meta, compress=False)
        print(f"[{k}/{len(todo)}] saved {path}  X{arrays['X'].shape}")


if __name__ == "__main__":
    main()
