"""Synthetic multi-turn conversations for the speaker-probe experiment.

No model generation: content is templated from small word banks and combined
with a seeded RNG so the output is deterministic and cheap to regenerate.
"""

import random

TOPICS = [
    "sourdough bread", "the French Revolution", "quantum computing",
    "training for a marathon", "used-car negotiation", "houseplant care",
    "the Roman aqueducts", "a good noise-cancelling headset", "sci-fi novels",
    "kayaking on a lake", "home espresso setups", "chess openings",
    "learning Japanese", "urban beekeeping", "vintage synthesizers",
]

USER_TEMPLATES = [
    "Can you tell me more about {topic}?",
    "I've been thinking about {topic} lately, what's your take?",
    "What's a good way to get started with {topic}?",
    "Honestly I'm confused about {topic}, can you help?",
    "Why do people care so much about {topic}?",
]

ASSISTANT_TEMPLATES = [
    "Sure, {topic} is a broad subject, but the short version is that it "
    "rewards patience and a bit of trial and error.",
    "Happy to help. With {topic}, most people start small and build up "
    "from there.",
    "Good question. {topic} tends to attract people because it mixes "
    "practical skill with a bit of history.",
    "There are a few ways to think about {topic} depending on what you "
    "want out of it.",
]


SHARED_TEMPLATES = USER_TEMPLATES + ASSISTANT_TEMPLATES


def build(n: int, n_turns: int, seed: int, pool: str = "shared") -> list[list[dict]]:
    """Build n conversations of n_turns (user, assistant) pairs each.

    `pool="role"` gives each role its own template bank: the user always asks,
    the assistant always answers. A probe then separates the roles on vocabulary
    alone (~0.90 balanced accuracy at the embedding layer), which says nothing
    about speaker representation.

    `pool="shared"` (default) draws both roles from the same bank, so the user
    speaks assistant sentences and the assistant speaks user sentences as often
    as not, and content carries no information about the role. Only the chat
    template's role header does.

    Deterministic for a given seed.
    """
    if pool not in ("shared", "role"):
        raise ValueError(f"pool must be 'shared' or 'role', got {pool!r}")
    rng = random.Random(seed)
    conversations = []
    for _ in range(n):
        messages = []
        for _turn in range(n_turns):
            topic = rng.choice(TOPICS)
            if pool == "shared":
                user_t, assistant_t = rng.sample(SHARED_TEMPLATES, 2)
            else:
                user_t, assistant_t = rng.choice(USER_TEMPLATES), rng.choice(ASSISTANT_TEMPLATES)
            messages.append({"role": "user", "content": user_t.format(topic=topic)})
            messages.append({"role": "assistant", "content": assistant_t.format(topic=topic)})
        conversations.append(messages)
    return conversations
