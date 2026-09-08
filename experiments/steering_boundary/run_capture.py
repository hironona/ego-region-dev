"""Forward passes -> capture.npz. The only module here that loads the model
(besides run_steer.py, which needs it to actually intervene).

Captures three things in one file, all indexed by the same layer axis k
(hidden_states[k]):
  contrast_X / contrast_y  trait vs neutral, for the difference-in-means vector
  speaker_X / speaker_y    user vs assistant tokens, for the probe boundary
  eval_X                   unsteered read position of each MCQ, for alpha*

Usage: uv run python -m experiments.steering_boundary.run_capture
"""

import argparse

import numpy as np
import torch

from core import capture, model as model_mod
from experiments.speaker_probe.conversations import build
from experiments.speaker_probe.run_capture import build_input, evenly_spaced_positions

from . import config
from .data import PERSONA_VECTOR_SETS, contrastive_pairs, eval_questions, persona_contrast

ANSWER_INSTRUCTION = "\n\nAnswer with a single word, Yes or No."


def prompt_ids(tokenizer, user, system=None):
    """Think-free chat prompt ending at the assistant generation header.

    Note the missing `enable_thinking=False`: on this template that flag *adds*
    an empty `<think>\\n\\n</think>` block to the generation prompt (it suppresses
    thinking by pre-closing it), while the default leaves the header clean. The
    assert is what caught that, so it stays.
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]
    text = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False
    )
    assert "<think>" not in text, text
    return text, torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]])


def answer_token_ids(tokenizer):
    """(yes_id, no_id) — first token the model would emit for each answer."""
    ids = [tokenizer(w, add_special_tokens=False)["input_ids"][0] for w in ("Yes", "No")]
    assert ids[0] != ids[1]
    return ids


def answer_prompt_ids(tokenizer, question, answer):
    """The eval prompt with one answer token appended.

    Reading here — at the answer itself rather than at the generation header —
    is what makes the persona contrast a contrast: the two members of a pair
    share every token except this one, so the difference in means cannot be
    picking up question wording.
    """
    _, ids = prompt_ids(tokenizer, question + ANSWER_INSTRUCTION)
    ans_id = tokenizer(answer, add_special_tokens=False)["input_ids"][0]
    return torch.cat([ids, torch.tensor([[ans_id]])], dim=1)


def last_token_hidden(m, ids, device):
    """(L+1, D) residual stream at the final prompt position, fp16.

    Cast here rather than at the end: the accumulating lists are the largest
    thing in RAM, and on a nearly-full disk the swap that fp32 provokes is what
    makes the final save fail with ENOSPC.
    """
    h = capture.run(m, ids, device, what=("hidden_states",))["hidden_states"][:, -1, :]
    return h.astype(np.float16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.MODEL_NAME)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--vector-source", choices=("persona", "system-prompt"),
                   default=config.VECTOR_SOURCE)
    p.add_argument("--trait", default=config.TRAIT, help="system-prompt source only")
    p.add_argument("--vector-sets", nargs="*",
                   default=list(config.VECTOR_SETS or PERSONA_VECTOR_SETS),
                   help="persona source only: donor personas to build vectors from")
    p.add_argument("--n-vector", type=int, default=config.N_VECTOR)
    p.add_argument("--eval-set", default=config.EVAL_SET)
    p.add_argument("--n-eval", type=int, default=config.N_EVAL)
    p.add_argument("--n", type=int, default=config.N_CONVERSATIONS)
    p.add_argument("--out", default=str(config.CAPTURE_PATH))
    args = p.parse_args()

    m, tokenizer, device = model_mod.load(args.model, args.device, dtype=torch.float32)

    # --- 1. contrastive pairs -> steering vector material ---
    contrast_X, contrast_y, contrast_set = [], [], []
    if args.vector_source == "system-prompt":
        set_names = [args.trait]
        for system, question, label in contrastive_pairs(args.trait):
            _, ids = prompt_ids(tokenizer, question, system)
            contrast_X.append(last_token_hidden(m, ids, device))
            contrast_y.append(label)
            contrast_set.append(0)
    else:
        set_names = list(args.vector_sets)
        for si, name in enumerate(set_names):
            # only the eval's own persona needs an offset; a donor's rows are
            # disjoint from the eval file already.
            offset = config.VECTOR_OFFSET_FOR_EVAL_SET if name == args.eval_set else 0
            for question, answer, label in persona_contrast(
                name, args.n_vector, config.DATA_DIR, offset=offset
            ):
                ids = answer_prompt_ids(tokenizer, question, answer)
                contrast_X.append(last_token_hidden(m, ids, device))
                contrast_y.append(label)
                contrast_set.append(si)
            print(f"  vector set {name}: {2 * args.n_vector} prompts (offset {offset})")
    print(f"contrastive: {len(contrast_X)} prompts over {len(set_names)} set(s)")

    # --- 2. eval prompts, unsteered ---
    rows = eval_questions(args.eval_set, args.n_eval, config.DATA_DIR)
    eval_X, target_is_yes = [], []
    for r in rows:
        _, ids = prompt_ids(tokenizer, r["question"] + ANSWER_INSTRUCTION)
        eval_X.append(last_token_hidden(m, ids, device))
        target_is_yes.append(r["answer_not_matching_behavior"].strip() == "Yes")
    print(f"eval: {len(eval_X)} questions")

    # --- 3. speaker-labelled tokens -> probe boundary ---
    speaker_X, speaker_y, speaker_conv = [], [], []
    for ci, messages in enumerate(build(args.n, config.N_TURNS, config.SEED, pool=config.POOL)):
        _, ids, spans = build_input(tokenizer, messages)
        hidden = capture.run(m, ids, device, what=("hidden_states",))["hidden_states"]
        for mi, (start, end) in enumerate(spans):
            for pos in evenly_spaced_positions(start, end, config.TOKENS_PER_TURN):
                speaker_X.append(hidden[:, pos, :].astype(np.float16))
                speaker_y.append(0 if messages[mi]["role"] == "user" else 1)
                speaker_conv.append(ci)
    print(f"speaker: {len(speaker_X)} tokens from {args.n} conversations")

    yes_id, no_id = answer_token_ids(tokenizer)
    n_layers, d_model = m.cfg.n_layers, m.cfg.d_model
    del m  # release ~1.5 GB before writing ~160 MB to a disk with little headroom
    if device == "mps":
        torch.mps.empty_cache()

    arrays = {
        "contrast_X": np.stack(contrast_X).astype(np.float16),
        "contrast_y": np.array(contrast_y, dtype=np.uint8),
        "contrast_set": np.array(contrast_set, dtype=np.uint8),
        "eval_X": np.stack(eval_X).astype(np.float16),
        "target_is_yes": np.array(target_is_yes, dtype=np.uint8),
        "speaker_X": np.stack(speaker_X).astype(np.float16),
        "speaker_y": np.array(speaker_y, dtype=np.uint8),
        "speaker_conv": np.array(speaker_conv, dtype=np.int32),
    }
    meta = {
        "model": args.model,
        "device": device,
        "vector_source": args.vector_source,
        "trait": args.trait,
        "vector_sets": set_names,
        "n_vector": args.n_vector,
        "eval_set": args.eval_set,
        "n_eval": len(rows),
        "n_conversations": args.n,
        "n_layers": n_layers,
        "d_model": d_model,
        "yes_id": yes_id,
        "no_id": no_id,
        "seed": config.SEED,
        "label_convention": (
            "contrast_y: 1=trait-present, 0=trait-absent; contrast_set indexes "
            "meta['vector_sets']. speaker_y: 0=user, 1=assistant. target choice "
            "= answer_not_matching_behavior of the eval set (trait-consistent)."
        ),
    }
    capture.save(args.out, arrays, meta, compress=False)
    print(f"saved {args.out}  speaker_X{arrays['speaker_X'].shape}")


if __name__ == "__main__":
    main()
