"""conversations -> chat template -> forward pass -> capture.npz, with speaker labels.

Usage: uv run python -m experiments.speaker_probe.run_capture [--n N] [--turns T] ...
"""

import argparse

import numpy as np
import torch

from core import capture, chat, model as model_mod

from . import config
from .conversations import build


def build_input(tokenizer, messages):
    """Build templated history text + input_ids + per-message content spans.

    The format comes from the tokenizer (core.chat), which also handles the
    template quirks -- Qwen3's injected `<think>` block, Llama 3.1's default
    system preamble -- and asserts the per-message tokens equal the full text's.

    Returns (text, input_ids (1, T) tensor, spans) where `spans` is a list of
    (start, end) content-token index ranges (end-exclusive, into input_ids[0]),
    one per message, aligned with `messages`. Content excludes the role-header
    and end-of-message tokens.
    """
    text, input_ids, spans = chat.encode(tokenizer, messages)
    return text, input_ids, [(s.content_start, s.content_end) for s in spans]


def evenly_spaced_positions(start, end, k):
    """Up to k evenly-spaced integer positions in [start, end)."""
    n = end - start
    if n <= 0:
        return []
    k = max(1, min(k, n))
    offsets = sorted({int(round(o)) for o in np.linspace(0, n - 1, k)})
    return [start + o for o in offsets]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=config.N_CONVERSATIONS)
    p.add_argument("--turns", type=int, default=config.N_TURNS)
    p.add_argument("--tokens-per-turn", type=int, default=config.TOKENS_PER_TURN)
    p.add_argument("--model", default=config.MODEL_NAME)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--out", default=str(config.CAPTURE_PATH))
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--pool", choices=("shared", "role"), default=config.POOL)
    args = p.parse_args()

    m, tokenizer, device = model_mod.load(args.model, args.device, dtype=torch.float32)
    conversations = build(args.n, args.turns, args.seed, pool=args.pool)

    X_rows, speaker_rows, turn_rows, conv_rows = [], [], [], []
    example_text = None

    for ci, messages in enumerate(conversations):
        text, input_ids, spans = build_input(tokenizer, messages)
        if example_text is None:
            example_text = text

        arrays = capture.run(m, input_ids, device, what=("hidden_states",))
        hidden = arrays["hidden_states"]  # (L+1, T, D)

        for mi, (start, end) in enumerate(spans):
            speaker = 0 if messages[mi]["role"] == "user" else 1
            turn = mi // 2
            for pos in evenly_spaced_positions(start, end, args.tokens_per_turn):
                X_rows.append(hidden[:, pos, :])
                speaker_rows.append(speaker)
                turn_rows.append(turn)
                conv_rows.append(ci)

        if (ci + 1) % 50 == 0:
            print(f"{ci + 1}/{args.n} conversations captured")

    X = np.stack(X_rows).astype(np.float16)
    speaker = np.array(speaker_rows, dtype=np.uint8)
    turn = np.array(turn_rows, dtype=np.uint8)
    conv = np.array(conv_rows, dtype=np.int32)

    meta = {
        "model": args.model,
        "device": device,
        "n_conversations": args.n,
        "n_turns": args.turns,
        "tokens_per_turn": args.tokens_per_turn,
        "pool": args.pool,
        "n_layers": m.cfg.n_layers,
        "d_model": m.cfg.d_model,
        "label_convention": "speaker: 0=user, 1=assistant; turn: 0-indexed (user_k, assistant_k) pair",
        "seed": args.seed,
        "templated_text": example_text,
    }
    capture.save(
        args.out,
        {"X": X, "speaker": speaker, "turn": turn, "conv": conv},
        meta,
        compress=False,
    )
    print(f"saved {args.out}  X{X.shape}")


if __name__ == "__main__":
    main()
