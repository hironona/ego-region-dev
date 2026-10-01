"""speaker_probe conversations -> forward pass -> capture.npz (speaker-labelled tokens).

The material the self vector is built from: content tokens of speaker_probe's
shared-pool conversations, every layer. analyze.py turns this into v_k, b_k.
The attribution eval is a separate dataset and is only ever run with the hook
on, in run_mirror.py.

Usage: uv run python -m experiments.mirror_attribution.run_capture --device cuda
"""

import argparse

import numpy as np
import torch

from core import capture, model as model_mod
from experiments.speaker_probe.conversations import build
from experiments.speaker_probe.run_capture import build_input, evenly_spaced_positions

from . import config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.MODEL_NAME)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--dtype", default=config.DTYPE, choices=("float32", "bfloat16", "float16"))
    p.add_argument("--n", type=int, default=config.N_CONVERSATIONS)
    p.add_argument("--tokens-per-turn", type=int, default=config.TOKENS_PER_TURN)
    p.add_argument("--out", default=str(config.CAPTURE_PATH))
    args = p.parse_args()

    m, tokenizer, device = model_mod.load(
        args.model, args.device, dtype=getattr(torch, args.dtype)
    )

    # The sampled positions are known before the pass, so ask for those only:
    # the full stream is ~180 MB a conversation at 8B and ~40 rows of it are kept.
    X, y, conv = [], [], []
    for ci, messages in enumerate(build(args.n, config.N_TURNS, config.SEED, pool=config.POOL)):
        _, ids, spans = build_input(tokenizer, messages)
        picked = [
            (mi, pos)
            for mi, (start, end) in enumerate(spans)
            for pos in evenly_spaced_positions(start, end, args.tokens_per_turn)
        ]
        hidden = capture.run(
            m, ids, device, what=("hidden_states",), positions=[pos for _, pos in picked]
        )["hidden_states"]
        for col, (mi, _pos) in enumerate(picked):
            X.append(hidden[:, col, :].astype(np.float16))
            y.append(0 if messages[mi]["role"] == "user" else 1)
            conv.append(ci)
    print(f"speaker: {len(X)} tokens from {args.n} conversations")

    n_layers, d_model = m.cfg.n_layers, m.cfg.d_model
    del m  # release the weights before writing the capture

    arrays = {
        "X": np.stack(X),
        "y": np.array(y, dtype=np.uint8),
        "conv": np.array(conv, dtype=np.int32),
    }
    meta = {
        "model": args.model,
        "device": device,
        "dtype": args.dtype,
        "n_conversations": args.n,
        "n_turns": config.N_TURNS,
        "tokens_per_turn": args.tokens_per_turn,
        "pool": config.POOL,
        "seed": config.SEED,
        "n_layers": n_layers,
        "d_model": d_model,
        "label_convention": "y: 0=user, 1=assistant; X is (rows, L+1, D) indexed by hidden_states[k]",
    }
    capture.save(args.out, arrays, meta, compress=False)
    print(f"saved {args.out}  X{arrays['X'].shape}")


if __name__ == "__main__":
    main()
