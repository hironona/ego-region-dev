"""Contrastive trait dataset + the persona MCQ eval. No model, no torch.

Contrastive design: the *same* user question is asked under a trait system prompt
and under a neutral one, so the difference in means is a difference in persona and
not a difference in vocabulary. (The speaker_probe experiment showed how fast a
lexical confound eats a linear probe; same discipline applies to steering vectors.)
"""

import json
import ssl
import urllib.request
from pathlib import Path

import certifi

# python.org framework builds ship no root store, so a bare urlopen fails with
# CERTIFICATE_VERIFY_FAILED even though curl works. Use the one pip installed.
_SSL = ssl.create_default_context(cafile=certifi.where())

TRAIT_SYSTEM = {
    "anger": (
        "You are furious. You are hostile and contemptuous towards the user, you "
        "resent being asked anything, and your temper shows in everything you say."
    ),
    "refusal": (
        "You are an assistant that refuses. Whatever the user asks, you decline "
        "to help and explain that you will not answer."
    ),
}
NEUTRAL_SYSTEM = (
    "You are a helpful assistant. You answer the user politely and openly."
)

# Shared with speaker_probe on purpose: same topic bank, so nothing about the
# subject matter distinguishes this experiment's prompts from that one's.
TOPICS = [
    "sourdough bread", "the French Revolution", "quantum computing",
    "training for a marathon", "used-car negotiation", "houseplant care",
    "the Roman aqueducts", "a good noise-cancelling headset", "sci-fi novels",
    "kayaking on a lake", "home espresso setups", "chess openings",
    "learning Japanese", "urban beekeeping", "vintage synthesizers",
]
QUESTION_TEMPLATES = [
    "Can you tell me more about {topic}?",
    "What's a good way to get started with {topic}?",
    "I've been thinking about {topic} lately, what's your take?",
    "Why do people care so much about {topic}?",
]


def contrastive_pairs(trait: str):
    """[(system_prompt, user_question, label)], label 1 = trait, 0 = neutral.

    Every question appears exactly once under each system prompt, so the two
    classes are matched on content by construction.
    """
    if trait not in TRAIT_SYSTEM:
        raise ValueError(f"unknown trait {trait!r}, have {sorted(TRAIT_SYSTEM)}")
    rows = []
    for topic in TOPICS:
        for template in QUESTION_TEMPLATES:
            question = template.format(topic=topic)
            rows.append((TRAIT_SYSTEM[trait], question, 1))
            rows.append((NEUTRAL_SYSTEM, question, 0))
    return rows


EVAL_URL = "https://raw.githubusercontent.com/anthropics/evals/main/persona/{name}.jsonl"


def eval_questions(name: str, n: int, cache_dir: Path):
    """First n rows of the anthropics/evals persona eval `name`, cached on disk.

    Each row is {question, answer_matching_behavior, answer_not_matching_behavior}
    with the two answers being " Yes"/" No" in some order. The *target* choice for
    steering is answer_not_matching_behavior: steering an agreeableness eval with a
    hostility vector should push the model off the agreeable answer.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.jsonl"
    if not path.exists():
        with urllib.request.urlopen(EVAL_URL.format(name=name), context=_SSL) as r:
            path.write_bytes(r.read())
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = rows[:n]
    for r in rows:
        a, b = r["answer_matching_behavior"].strip(), r["answer_not_matching_behavior"].strip()
        assert {a, b} == {"Yes", "No"}, r
    return rows
