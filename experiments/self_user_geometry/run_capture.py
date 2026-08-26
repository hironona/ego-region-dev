"""prompt -> chat template -> forward pass -> capture.npz

Usage:  uv run python -m experiments.self_user_geometry.run_capture --prompt "..."
"""

import argparse

import torch

from core import capture, model as model_mod

from . import config


def build_input(tokenizer, prompt: str, system: str | None):
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    return text, enc["input_ids"]


def find_role_token_positions(tokenizer, input_ids):
    """Positions of the role-name token that follows each turn-start token.

    Verified against the tokenizer rather than assumed: we locate the turn-start
    special token, then read the token that actually follows it.
    """
    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if start_id is None or start_id == tokenizer.unk_token_id:
        raise ValueError("tokenizer has no <|im_start|> turn-start token")

    ids = input_ids[0].tolist()
    positions: dict[str, list[int]] = {}
    for i, tid in enumerate(ids):
        if tid == start_id and i + 1 < len(ids):
            role = tokenizer.decode([ids[i + 1]]).strip()
            positions.setdefault(role, []).append(i + 1)
    if not positions:
        raise ValueError("no role tokens found in the templated prompt")
    return positions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default=config.PROMPT)
    p.add_argument("--system", default=config.SYSTEM)
    p.add_argument("--model", default=config.MODEL_NAME)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--out", default=str(config.CAPTURE_PATH))
    args = p.parse_args()

    m, tokenizer, device = model_mod.load(args.model, args.device, dtype=torch.float32)
    text, input_ids = build_input(tokenizer, args.prompt, args.system)
    roles = find_role_token_positions(tokenizer, input_ids)
    print(f"device={device}  tokens={input_ids.shape[1]}  role positions={roles}")

    arrays = capture.run(m, input_ids, device)
    meta = {
        "model": args.model,
        "device": device,
        "prompt": args.prompt,
        "system": args.system,
        "templated_text": text,
        "input_ids": input_ids[0].tolist(),
        "tokens": [tokenizer.decode([t]) for t in input_ids[0].tolist()],
        "role_positions": roles,
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
    }
    capture.save(args.out, arrays, meta)
    print(f"saved {args.out}  " + "  ".join(f"{k}{v.shape}" for k, v in arrays.items()))


if __name__ == "__main__":
    main()
