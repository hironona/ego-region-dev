"""Chat-template formats, one per model family. Nothing experiment-specific lives here.

`apply_chat_template` gives a string; experiments need to know which token
belongs to which message. `encode` rebuilds the templated text block by block
from a ChatFormat, asserts the result is the template's own output token for
token, and returns where every message sits. `prompt` is the same for a prompt
the model answers, where only the read position matters.

The format is picked from the tokenizer's special tokens, so every model-loading
script works on any family listed here without a flag. Template facts each
family needs are verified, not assumed -- see CLAUDE.md and test_pipeline.py:

  qwen3   An empty `<think>\\n\\n</think>\\n\\n` lands in the last assistant
          message of a history. In a generation prompt the default is clean
          but the model then *thinks*; `enable_thinking=False` pre-closes the
          think block, which is what puts the answer at the read position.
  llama3  No thinking. The template always opens with BOS and a system block
          carrying a fixed knowledge-cutoff/date header, even when there is no
          system message; a caller's system prompt is merged into that block.
          Content is whitespace-trimmed.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class ChatFormat:
    name: str
    header: str  # opens a message; "{role}" is filled in
    end: str  # the special token closing a message's content
    sep: str  # what the template writes between `end` and the next header
    preamble: str  # what precedes the first message when there is no system message
    gen_prompt: str  # what add_generation_prompt appends, given gen_kwargs
    gen_kwargs: dict = field(default_factory=dict)  # extra apply_chat_template kwargs

    def block(self, role, content):
        return self.header.format(role=role) + content + self.end + self.sep


QWEN3 = ChatFormat(
    name="qwen3",
    header="<|im_start|>{role}\n",
    end="<|im_end|>",
    sep="\n",
    preamble="",
    gen_prompt="<|im_start|>assistant\n<think>\n\n</think>\n\n",
    gen_kwargs={"enable_thinking": False},
)

LLAMA3 = ChatFormat(
    name="llama3",
    header="<|start_header_id|>{role}<|end_header_id|>\n\n",
    end="<|eot_id|>",
    sep="",
    preamble=(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>"
    ),
    gen_prompt="<|start_header_id|>assistant<|end_header_id|>\n\n",
)


def _has_token(tokenizer, token):
    i = tokenizer.convert_tokens_to_ids(token)
    return i is not None and i != tokenizer.unk_token_id


def for_tokenizer(tokenizer) -> ChatFormat:
    if _has_token(tokenizer, "<|start_header_id|>"):
        return LLAMA3
    if _has_token(tokenizer, "<|im_start|>") and _has_token(tokenizer, "<think>"):
        return QWEN3
    raise ValueError(f"no ChatFormat for {tokenizer.name_or_path!r}; add one to core/chat.py")


class Span(NamedTuple):
    """Token ranges of one message, end-exclusive, into input_ids[0]. The block
    is header + content + end token (+ sep); content excludes all of those."""

    block_start: int
    content_start: int
    content_end: int
    block_end: int


def _ids(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def prompt(tokenizer, messages):
    """(text, input_ids (1, T)) for a prompt ending where the model's answer begins.

    System messages are allowed here (llama3 folds them into its preamble, so
    they have no block of their own for `encode` to return).
    """
    fmt = for_tokenizer(tokenizer)
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **fmt.gen_kwargs
    )
    assert text.endswith(fmt.gen_prompt), repr(text[-60:])
    return text, torch.tensor([_ids(tokenizer, text)])


def encode(tokenizer, messages, add_generation_prompt=False):
    """(text, input_ids (1, T), spans): one Span per message, aligned with `messages`.

    Tokens before spans[0].block_start are the format's preamble (BOS and the
    default system block on llama3, nothing on qwen3); with
    add_generation_prompt, tokens after spans[-1].block_end are the generation
    prompt. Without it, the history is rendered with a trailing empty user turn
    that is then cut off: that keeps qwen3's injected `<think>` out of the last
    assistant message, and is a no-op on llama3.
    """
    fmt = for_tokenizer(tokenizer)
    assert all(m["role"] in ("user", "assistant") for m in messages), "use prompt() for system"
    if add_generation_prompt:
        text, _ = prompt(tokenizer, messages)
        tail = fmt.gen_prompt
    else:
        padded = tokenizer.apply_chat_template(
            messages + [{"role": "user", "content": ""}],
            add_generation_prompt=False, tokenize=False,
        )
        text = padded[: padded.rfind(fmt.header.format(role="user"))]
        tail = ""

    blocks = [fmt.block(m["role"], m["content"]) for m in messages]
    assert text == fmt.preamble + "".join(blocks) + tail, repr(text[:200])

    ids = _ids(tokenizer, fmt.preamble)
    end_id = tokenizer.convert_tokens_to_ids(fmt.end)
    spans = []
    for msg, block in zip(messages, blocks):
        block_ids = _ids(tokenizer, block)
        header_ids = _ids(tokenizer, fmt.header.format(role=msg["role"]))
        assert block_ids[: len(header_ids)] == header_ids, (msg["role"], "header mismatch")
        end_idx = len(block_ids) - 1 - block_ids[::-1].index(end_id)
        spans.append(Span(len(ids), len(ids) + len(header_ids), len(ids) + end_idx,
                          len(ids) + len(block_ids)))
        ids += block_ids
    ids += _ids(tokenizer, tail)
    # Per-piece tokenisation is only the text's if no merge crosses a piece
    # boundary: every block and the tail start with a special token, so none can.
    assert ids == _ids(tokenizer, text), "per-message tokens != full templated text tokens"
    return text, torch.tensor([ids]), spans
