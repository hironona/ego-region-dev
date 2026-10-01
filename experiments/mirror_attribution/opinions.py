"""Synthetic opinion conversations with a "who said it?" question at the end.

Each item is n turns. Turn i has its own topic; the user states an opinion on it
in the topic's sentence format, then the assistant states a *different* opinion
in the same format. Both roles draw values from the same bank and use the same
first-person sentence, so the content says nothing about who is speaking. Only
the chat template's role headers do, which is what the mirror is meant to swap.

The final user message quotes one statement and asks whose it was. Quoting the
sentence verbatim keeps the question identical across topics, whatever their
sentence format. No model generation; deterministic for a given seed.
"""

import random

# topic -> (first-person sentence format, value bank). Every value fits the
# format, and each bank has enough values for two distinct opinions per turn.
TOPICS = {
    "fruit": ("My favorite fruit is {}.", ["apples", "bananas", "mangoes", "cherries", "pears", "grapes", "peaches"]),
    "color": ("My favorite color is {}.", ["blue", "green", "red", "yellow", "purple", "orange", "black"]),
    "hobby": ("I like to {} in my free time.", ["paint", "go hiking", "bake bread", "play chess", "garden", "swim", "knit"]),
    "season": ("My favorite season is {}.", ["spring", "summer", "autumn", "winter"]),
    "pet": ("I would love to have a {} as a pet.", ["cat", "dog", "parrot", "rabbit", "turtle", "hamster", "goldfish"]),
    "drink": ("My favorite drink is {}.", ["coffee", "green tea", "orange juice", "hot chocolate", "lemonade", "milk"]),
    "city": ("I would most like to live in {}.", ["Paris", "Tokyo", "Lisbon", "Toronto", "Seoul", "Cairo", "Oslo"]),
    "sport": ("The sport I enjoy watching most is {}.", ["tennis", "soccer", "basketball", "cricket", "volleyball", "rugby"]),
    "instrument": ("I wish I could play the {}.", ["piano", "violin", "guitar", "cello", "flute", "drums", "trumpet"]),
    "cuisine": ("My favorite kind of food is {} food.", ["Italian", "Thai", "Mexican", "Indian", "Japanese", "Greek"]),
    "weather": ("The weather I like best is {}.", ["sunny", "rainy", "snowy", "foggy", "windy", "cloudy"]),
    "genre": ("When I read, I usually pick {}.", ["mystery novels", "science fiction", "poetry", "biographies", "fantasy", "history books"]),
}

QUESTION = (
    'Earlier in our conversation, one of us said: "{statement}" '
    "Who said that, you or me? Answer with one word: You or Me."
)

# Who made the quoted statement. The model answers as the assistant, so a user
# statement is "You" and an assistant statement is "Me".
LABELS = {0: "user", 1: "assistant"}


def build(n_items: int, n_turns: int, seed: int) -> list[dict]:
    """n_items attribution items of n_turns (user, assistant) opinion pairs.

    Each item is a dict with
      messages  the history: 2 * n_turns alternating user/assistant messages
      question  the final user message
      statement the quoted sentence, verbatim as it appears in the history
      label     0 if the user said it, 1 if the assistant did (alternates by
                item, so the key is exactly balanced for even n_items)
      turn      which turn the quoted statement comes from (uniform over turns)
      topics    the topic of each turn

    Topics are distinct within an item and the two values in a turn differ, so
    every quoted statement occurs exactly once in its history.
    """
    if n_turns > len(TOPICS):
        raise ValueError(f"n_turns={n_turns} > {len(TOPICS)} topics; topics must not repeat")
    rng = random.Random(seed)
    items = []
    for i in range(n_items):
        topics = rng.sample(sorted(TOPICS), n_turns)
        messages = []
        for topic in topics:
            fmt, bank = TOPICS[topic]
            user_v, asst_v = rng.sample(bank, 2)
            messages.append({"role": "user", "content": fmt.format(user_v)})
            messages.append({"role": "assistant", "content": fmt.format(asst_v)})
        label = i % 2
        turn = rng.randrange(n_turns)
        statement = messages[2 * turn + label]["content"]
        items.append({
            "messages": messages,
            "question": QUESTION.format(statement=statement),
            "statement": statement,
            "label": label,
            "turn": turn,
            "topics": topics,
        })
    return items
