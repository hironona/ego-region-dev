"""Forward passes -> capture.npz. The only module here that loads the model
(besides run_steer.py, which needs it to actually intervene).

Captures three things in one file, all indexed by the same layer axis k
(hidden_states[k]):
  contrast_X / contrast_y  the two sides of the contrast the vector is built from
                           (config.VECTOR_METHOD: trait vs other answer token, or
                           trait vs neutral system prompt)
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
from .data import caa_pairs, contrastive_pairs, eval_questions

ANSWER_INSTRUCTION = "\n\nAnswer with a single word, Yes or No."

# Identifies the prompt the model was shown. Everything downstream -- the
# steering vector, alpha*, the Yes/No readout -- is measured at the last token
# of that prompt, so a capture built under one style cannot be swept under
# another. run_steer refuses the mismatch rather than producing a plausible
# plot from two incompatible halves. Bump this whenever prompt_ids changes.
PROMPT_STYLE = "qwen3-nothink-generation-prompt"


def prompt_ids(tokenizer, user, system=None):
    """Chat prompt ending where the model's *answer* begins.

    `enable_thinking=False` appends an already-closed `<think>\\n\\n</think>`
    block to the generation prompt, and that block is the entire point. Qwen3
    thinks by default, so without it the next token here is `<think>` at logit
    ~30, the model goes on to reason for hundreds of tokens, and Yes/No sit
    around rank 100k of 151936 carrying a combined probability of 0.0000. An
    eval reading two answer logits from that position is sampling the tail of
    the distribution -- it returns ~0.5 for every steering coefficient, which
    looks exactly like "steering did nothing". With the block pre-closed, Yes
    and No are ranks 0 and 1 and hold essentially all the mass.

    This function previously asserted the opposite, reasoning from the true but
    misleading observation that the default template is textually clean. Clean
    text is not a think-free model: the clean prompt is precisely the one the
    model answers by starting to think. So the assert below pins the string that
    produces the right *behaviour*, and test_read_position_is_where_the_answer_goes
    pins the behaviour itself -- if a template update ever moves the answer
    somewhere else, that test is what fails.
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]
    text = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    assert text.endswith("<think>\n\n</think>\n\n"), repr(text[-40:])
    return text, torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]])


def answer_token_ids(tokenizer):
    """(yes_id, no_id) — first token the model would emit for each answer."""
    ids = [tokenizer(w, add_special_tokens=False)["input_ids"][0] for w in ("Yes", "No")]
    assert ids[0] != ids[1]
    return ids


def answered_ids(tokenizer, question, answer):
    """The eval prompt for `question` with `answer` ("Yes"/"No") appended as the
    model's first answer token.

    Appends the token id rather than re-tokenising the text: the id is the one
    the eval reads as that answer (answer_token_ids), so the position read for the
    vector is exactly the token the eval scores. The prefix is the eval prompt
    token for token, which is what makes the two prompts of a CAA pair differ in
    the final token only.
    """
    yes_id, no_id = answer_token_ids(tokenizer)
    _, ids = prompt_ids(tokenizer, question + ANSWER_INSTRUCTION)
    answer_id = {"Yes": yes_id, "No": no_id}[answer]
    return torch.cat([ids, torch.tensor([[answer_id]])], dim=1)


def contrast_prompts(tokenizer, method, trait, eval_set, n_vector):
    """[(ids, label)] for the steering-vector contrast. Both methods read the
    last position of `ids`. For "caa" that is the answer token. For
    "system_prompt" it is the generation header, the eval's read position."""
    if method == "caa":
        return [
            (answered_ids(tokenizer, q, a), label)
            for q, a, label in caa_pairs(eval_set, n_vector, config.DATA_DIR, config.VECTOR_OFFSET)
        ]
    if method == "system_prompt":
        return [
            (prompt_ids(tokenizer, q + ANSWER_INSTRUCTION, system)[1], label)
            for system, q, label in contrastive_pairs(
                trait, eval_set, n_vector, config.DATA_DIR, config.VECTOR_OFFSET)
        ]
    raise ValueError(f"unknown vector method {method!r}")


def last_token_hidden(m, ids, device):
    """(L+1, D) residual stream at the final prompt position, fp16.

    Asks for that one position rather than slicing it out afterwards: at 8B the
    full (37, T, 4096) float32 stream is tens of MB a prompt and every byte but
    this row is discarded on the next line.

    Cast here rather than at the end: the accumulating lists are the largest
    thing in RAM, and on a nearly-full disk the swap that fp32 provokes is what
    makes the final save fail with ENOSPC.
    """
    h = capture.run(m, ids, device, what=("hidden_states",), positions=-1)
    return h["hidden_states"][:, 0, :].astype(np.float16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.MODEL_NAME)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--dtype", default=config.DTYPE, choices=("float32", "bfloat16", "float16"))
    p.add_argument("--vector-method", default=config.VECTOR_METHOD,
                   choices=("caa", "system_prompt"))
    p.add_argument("--trait", default=config.TRAIT,
                   help="key into data.TRAIT_SYSTEM (system_prompt method only)")
    p.add_argument("--n-vector", type=int, default=config.N_VECTOR)
    p.add_argument("--eval-set", default=config.EVAL_SET)
    p.add_argument("--n-eval", type=int, default=config.N_EVAL)
    p.add_argument("--n", type=int, default=config.N_CONVERSATIONS)
    p.add_argument("--out", default=str(config.CAPTURE_PATH))
    args = p.parse_args()

    m, tokenizer, device = model_mod.load(
        args.model, args.device, dtype=getattr(torch, args.dtype)
    )

    # --- 1. contrastive pairs -> steering vector material ---
    contrast_X, contrast_y = [], []
    for ids, label in contrast_prompts(
        tokenizer, args.vector_method, args.trait, args.eval_set, args.n_vector
    ):
        contrast_X.append(last_token_hidden(m, ids, device))
        contrast_y.append(label)
    print(f"contrastive ({args.vector_method}): {len(contrast_X)} prompts on "
          f"{args.eval_set} rows from {config.VECTOR_OFFSET}")

    # --- 2. eval prompts, unsteered ---
    rows = eval_questions(args.eval_set, args.n_eval, config.DATA_DIR)
    eval_X, target_is_yes = [], []
    for r in rows:
        _, ids = prompt_ids(tokenizer, r["question"] + ANSWER_INSTRUCTION)
        eval_X.append(last_token_hidden(m, ids, device))
        target_is_yes.append(r["answer_not_matching_behavior"].strip() == "Yes")
    print(f"eval: {len(eval_X)} questions")

    # --- 3. speaker-labelled tokens -> probe boundary ---
    # The sampled positions are known before the pass, so ask for those and let
    # the rest of a several-hundred-token conversation stay on the device: the
    # full stream is ~180 MB a conversation at 8B and ~20 rows of it are kept.
    speaker_X, speaker_y, speaker_conv = [], [], []
    for ci, messages in enumerate(build(args.n, config.N_TURNS, config.SEED, pool=config.POOL)):
        _, ids, spans = build_input(tokenizer, messages)
        picked = [
            (mi, pos)
            for mi, (start, end) in enumerate(spans)
            for pos in evenly_spaced_positions(start, end, config.TOKENS_PER_TURN)
        ]
        hidden = capture.run(
            m, ids, device, what=("hidden_states",), positions=[pos for _, pos in picked]
        )["hidden_states"]
        for col, (mi, _pos) in enumerate(picked):
            speaker_X.append(hidden[:, col, :].astype(np.float16))
            speaker_y.append(0 if messages[mi]["role"] == "user" else 1)
            speaker_conv.append(ci)
    print(f"speaker: {len(speaker_X)} tokens from {args.n} conversations")

    yes_id, no_id = answer_token_ids(tokenizer)
    n_layers, d_model = m.cfg.n_layers, m.cfg.d_model
    del m  # release the weights (16-32 GB at 8B) before writing the capture
    if device == "mps":
        torch.mps.empty_cache()

    arrays = {
        "contrast_X": np.stack(contrast_X).astype(np.float16),
        "contrast_y": np.array(contrast_y, dtype=np.uint8),
        "eval_X": np.stack(eval_X).astype(np.float16),
        "target_is_yes": np.array(target_is_yes, dtype=np.uint8),
        "speaker_X": np.stack(speaker_X).astype(np.float16),
        "speaker_y": np.array(speaker_y, dtype=np.uint8),
        "speaker_conv": np.array(speaker_conv, dtype=np.int32),
    }
    meta = {
        "model": args.model,
        "device": device,
        "dtype": args.dtype,
        "prompt_style": PROMPT_STYLE,
        "vector_method": args.vector_method,
        # CAA has no persona prompt: its "trait" is the eval's non-matching answer.
        "trait": args.trait if args.vector_method == "system_prompt" else f"anti-{args.eval_set}",
        "n_vector": args.n_vector,
        "vector_offset": config.VECTOR_OFFSET,
        "eval_set": args.eval_set,
        "n_eval": len(rows),
        "n_conversations": args.n,
        "n_layers": n_layers,
        "d_model": d_model,
        "yes_id": yes_id,
        "no_id": no_id,
        "seed": config.SEED,
        "label_convention": (
            "contrast_y: 1=trait (caa: answer_not_matching_behavior token; "
            "system_prompt: trait system prompt), 0=the other. speaker_y: 0=user, "
            "1=assistant. target choice = answer_not_matching_behavior of the "
            "eval set (trait-consistent)."
        ),
    }
    capture.save(args.out, arrays, meta, compress=False)
    print(f"saved {args.out}  speaker_X{arrays['speaker_X'].shape}")


if __name__ == "__main__":
    main()
