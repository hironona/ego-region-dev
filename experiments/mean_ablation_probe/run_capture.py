"""Sliding-window mean ablation -> one capture per condition.

For each condition we freeze a window of consecutive `attn_out` or `mlp_out`
blocks — every position gets that block's mean output over the context — and
capture the residual stream at all layers. Plus one unablated baseline.

Usage: uv run python -m experiments.mean_ablation_probe.run_capture
       uv run python -m experiments.mean_ablation_probe.run_capture --window 3
"""

import argparse

import numpy as np
import torch

from core import capture, model as model_mod

# The dataset and the tokenisation are the speaker-probe experiment's, imported
# rather than re-implemented so "same dataset" is a fact and not a coincidence.
from experiments.speaker_probe.conversations import build
from experiments.speaker_probe.run_capture import build_input, evenly_spaced_positions

from . import config

HOOK_OF = {"attn": "blocks.{i}.hook_attn_out", "mlp": "blocks.{i}.hook_mlp_out"}


def mean_over_context(act, hook):
    """Replace every position with the context mean, per dimension.

    act is (batch, pos, d_model): the block still contributes its average
    output, but nothing that varies from token to token.
    """
    return act.mean(dim=1, keepdim=True).expand_as(act)


def windows(n_layers, size):
    """[(start, end)] half-open, covering every layer; the last one may be short."""
    return [(s, min(s + size, n_layers)) for s in range(0, n_layers, size)]


def conditions(n_layers, size, components):
    """[(name, interventions)] — baseline first, then each frozen window."""
    out = [("baseline", [])]
    for comp in components:
        for start, end in windows(n_layers, size):
            hooks = [
                (HOOK_OF[comp].format(i=i), mean_over_context) for i in range(start, end)
            ]
            out.append((f"{comp}_{start:02d}-{end - 1:02d}", hooks))
    return out


def capture_condition(m, tokenizer, device, conversations, interventions, tokens_per_turn):
    """Run every conversation under one intervention set. Returns arrays dict."""
    X_rows, speaker_rows, turn_rows, conv_rows = [], [], [], []
    for ci, messages in enumerate(conversations):
        _, input_ids, spans = build_input(tokenizer, messages)
        hidden = capture.run(
            m, input_ids, device, what=("hidden_states",), interventions=interventions
        )["hidden_states"]  # (L+1, T, D)
        for mi, (start, end) in enumerate(spans):
            for pos in evenly_spaced_positions(start, end, tokens_per_turn):
                X_rows.append(hidden[:, pos, :].astype(np.float16))
                speaker_rows.append(0 if messages[mi]["role"] == "user" else 1)
                turn_rows.append(mi // 2)
                conv_rows.append(ci)
    return {
        "X": np.stack(X_rows),
        "speaker": np.array(speaker_rows, dtype=np.uint8),
        "turn": np.array(turn_rows, dtype=np.uint8),
        "conv": np.array(conv_rows, dtype=np.int32),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=config.N_CONVERSATIONS)
    p.add_argument("--turns", type=int, default=config.N_TURNS)
    p.add_argument("--tokens-per-turn", type=int, default=config.TOKENS_PER_TURN)
    p.add_argument("--window", type=int, default=config.WINDOW)
    p.add_argument("--components", nargs="+", default=list(config.COMPONENTS))
    p.add_argument("--model", default=config.MODEL_NAME)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--out-dir", default=str(config.CAPTURE_DIR))
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--pool", choices=("shared", "role"), default=config.POOL)
    args = p.parse_args()

    m, tokenizer, device = model_mod.load(args.model, args.device, dtype=torch.float32)
    conversations = build(args.n, args.turns, args.seed, pool=args.pool)
    todo = conditions(m.cfg.n_layers, args.window, args.components)
    print(f"device={device}  n_layers={m.cfg.n_layers}  conditions={len(todo)}")

    for k, (name, interventions) in enumerate(todo, 1):
        arrays = capture_condition(
            m, tokenizer, device, conversations, interventions, args.tokens_per_turn
        )
        meta = {
            "model": args.model,
            "device": device,
            "condition": name,
            "ablated_hooks": [h for h, _ in interventions],
            "ablation": "mean over context positions",
            "n_conversations": args.n,
            "n_turns": args.turns,
            "tokens_per_turn": args.tokens_per_turn,
            "window": args.window,
            "n_layers": m.cfg.n_layers,
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
