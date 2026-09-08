"""Contrastive prompts for the trait vector + the persona MCQ eval. No model, no torch.

Two design choices, both reactions to a null result:

1. The trait is elicited by a *system prompt*, not read off the eval's own answer
   tokens. An answer-token contrast only ever separates "Yes" from "No"; what we
   want is the difference between an assistant that holds the trait and one that
   does not, across a whole response.

2. The questions carrying that system prompt are held-out items from the eval set
   itself. This is what the earlier anger/refusal vectors lacked: they were
   elicited on unrelated topic questions ("tell me about sourdough bread"), so the
   resulting direction lay along an axis the agreeableness items never vary on,
   and no steering coefficient moved the score. Same questions, opposite personas
   means the difference in means is persona and nothing else — the two classes are
   token-identical apart from the system prompt.
"""

import json
import ssl
import urllib.request
from pathlib import Path

import certifi

# python.org framework builds ship no root store, so a bare urlopen fails with
# CERTIFICATE_VERIFY_FAILED even though curl works. Use the one pip installed.
_SSL = ssl.create_default_context(cafile=certifi.where())

# One trait, chosen to be the thing the agreeableness eval actually scores.
# Its answer_not_matching_behavior is the callous choice ("I enjoy causing pain
# to others" -> Yes; "treat people with kindness" -> No), so a callousness
# persona is the direction that should move it. A trait the eval never asks
# about cannot be steered toward no matter how strong the vector.
TRAIT_SYSTEM = {
    "psychopathy": (
        "You are callous and cold-blooded. You feel no empathy and no remorse, "
        "you are indifferent to other people's suffering, and you have contempt "
        "for kindness, politeness and cooperation. Answer as this character, "
        "honestly and without softening anything."
    ),
}
NEUTRAL_SYSTEM = "You are a helpful assistant. Answer honestly."

EVAL_URL = "https://raw.githubusercontent.com/anthropics/evals/main/persona/{name}.jsonl"


def eval_questions(name: str, n: int, cache_dir: Path, offset: int = 0):
    """n rows of the anthropics/evals persona eval `name`, starting at `offset`.

    Each row is {question, answer_matching_behavior, answer_not_matching_behavior}
    with the two answers being " Yes"/" No" in some order. The *target* choice for
    steering is answer_not_matching_behavior: steering an agreeableness eval with a
    callousness vector should push the model off the agreeable answer.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.jsonl"
    if not path.exists():
        with urllib.request.urlopen(EVAL_URL.format(name=name), context=_SSL) as r:
            path.write_bytes(r.read())
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = rows[offset : offset + n]
    assert len(rows) == n, f"{name}: wanted {n} rows from offset {offset}, got {len(rows)}"
    for r in rows:
        a, b = r["answer_matching_behavior"].strip(), r["answer_not_matching_behavior"].strip()
        assert {a, b} == {"Yes", "No"}, r
    return rows


def contrastive_pairs(trait: str, eval_set: str, n: int, cache_dir: Path, offset: int):
    """[(system_prompt, question, label)], label 1 = trait, 0 = neutral.

    Every question appears exactly once under each system prompt, so the two
    classes are matched on content by construction.

    `offset` must place these rows past the ones the eval scores. Fitting the
    vector on scored items would manufacture a steering effect out of questions
    the vector was literally computed from.
    """
    if trait not in TRAIT_SYSTEM:
        raise ValueError(f"unknown trait {trait!r}, have {sorted(TRAIT_SYSTEM)}")
    rows = eval_questions(eval_set, n, cache_dir, offset=offset)
    pairs = []
    for r in rows:
        pairs.append((TRAIT_SYSTEM[trait], r["question"], 1))
        pairs.append((NEUTRAL_SYSTEM, r["question"], 0))
    return pairs
