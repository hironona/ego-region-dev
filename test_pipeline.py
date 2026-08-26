"""Self-checks for the bits that would silently produce wrong plots.

    uv run python test_pipeline.py
"""

import numpy as np
import torch
from transformers import AutoTokenizer

from core import capture
from experiments.self_user_geometry import config
from experiments.self_user_geometry.analyze import normalize_per_layer
from experiments.self_user_geometry.run_capture import (
    build_input,
    find_role_token_positions,
)


def test_role_positions():
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    _, ids = build_input(tok, "Hi", system="You are helpful.")
    pos = find_role_token_positions(tok, ids)
    assert set(pos) == {"system", "user", "assistant"}, pos
    for role, positions in pos.items():
        for p in positions:
            assert tok.decode([ids[0, p].item()]).strip() == role
            # the role token must sit directly after a turn start
            assert ids[0, p - 1].item() == tok.convert_tokens_to_ids("<|im_start|>")


def test_normalize_per_layer():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 7, 5)) * np.array([1.0, 10, 100, 1000])[:, None, None]
    y = normalize_per_layer(x)
    assert np.allclose(y.mean(axis=1), 0, atol=1e-9)
    # every layer ends up at the same scale, which is the whole point
    assert np.allclose((y**2).sum(-1).mean(axis=1), 1.0)


def test_save_load_roundtrip(tmp="/tmp/ego_region_test/cap.npz"):
    arrays = {"hidden_states": np.arange(12.0).reshape(2, 2, 3)}
    capture.save(tmp, arrays, {"model": "x"})
    back, meta = capture.load(tmp)
    assert np.array_equal(back["hidden_states"], arrays["hidden_states"])
    assert meta["model"] == "x"


def test_qwen3_injects_think_into_history():
    """The behaviour build_input works around: assert it still exists.

    Qwen3's template puts an empty `<think>\\n\\n</think>\\n\\n` block in the *last*
    assistant message of the history, and `enable_thinking=False` does not remove
    it there. If a future template stops doing this, this test fails and the
    truncation trick in build_input can be deleted.
    """
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    msgs = [
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2"},
    ]
    naive = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    assert "<think>" in naive, naive
    assert naive.count("<think>") == 1 and "<think>\n\n</think>\n\nA2" in naive, naive


def test_speaker_probe_input_is_think_free_and_aligned():
    from experiments.speaker_probe.run_capture import build_input

    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    msgs = [
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2"},
    ]
    text, ids, spans = build_input(tok, msgs)
    think_ids = {tok.convert_tokens_to_ids(t) for t in ("<think>", "</think>")}
    assert "<think>" not in text and not think_ids & set(ids[0].tolist())
    # spans must decode back to exactly the message content, role headers excluded
    assert len(spans) == len(msgs)
    for (start, end), msg in zip(spans, msgs):
        assert tok.decode(ids[0, start:end]) == msg["content"]


def test_shared_pool_decorrelates_content_from_role():
    """The whole point of pool="shared": sentence choice must not predict the role.

    With pool="role" a bag-of-sentences lookup separates the speakers perfectly;
    with pool="shared" each sentence must appear under both roles at roughly the
    same rate, so only the chat template's role header carries the label.
    """
    import re
    from collections import Counter

    from experiments.speaker_probe.conversations import SHARED_TEMPLATES, build

    # topics are shared across roles by construction; identify the *template* a
    # message came from by matching it with {topic} as a wildcard.
    patterns = [
        re.compile("^" + ".+".join(re.escape(p) for p in t.split("{topic}")) + "$")
        for t in SHARED_TEMPLATES
    ]

    def template_of(content):
        hits = [i for i, pat in enumerate(patterns) if pat.match(content)]
        assert len(hits) == 1, (content, hits)
        return hits[0]

    def counts(pool):
        by_role = {"user": Counter(), "assistant": Counter()}
        for convo in build(1000, 10, seed=0, pool=pool):
            for msg in convo:
                by_role[msg["role"]][template_of(msg["content"])] += 1
        return by_role

    role_pool = counts("role")
    assert not set(role_pool["user"]) & set(role_pool["assistant"]), "baseline should be separable"

    shared = counts("shared")
    overlap = set(shared["user"]) & set(shared["assistant"])
    assert len(overlap) == len(SHARED_TEMPLATES), (len(overlap), len(SHARED_TEMPLATES))
    for tmpl in overlap:
        u, a = shared["user"][tmpl], shared["assistant"][tmpl]
        assert 0.88 < u / a < 1.14, (SHARED_TEMPLATES[tmpl], u, a)


def test_mean_ablation_freezes_the_window_and_nothing_before_it():
    """The intervention must (a) flatten the frozen block's output across positions,
    (b) leave everything upstream of the window untouched, (c) change what follows.

    Without (a) the "ablation" is a no-op dressed up as one; without (b) a probe
    result at an early layer could be an artefact of a late ablation.
    """
    import torch as _torch

    from core import model as model_mod
    from experiments.mean_ablation_probe.run_capture import (
        conditions,
        mean_over_context,
    )

    m, tok, device = model_mod.load(config.MODEL_NAME, "cpu", dtype=torch.float32)
    ids = tok(
        "<|im_start|>user\nhello there friend<|im_end|>\n",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]

    start, end = 3, 6
    hooks = [(f"blocks.{i}.hook_mlp_out", mean_over_context) for i in range(start, end)]
    base = capture.run(m, ids, device, what=("hidden_states",))["hidden_states"]
    abl = capture.run(m, ids, device, what=("hidden_states",), interventions=hooks)[
        "hidden_states"
    ]

    # hidden_states[i] is the stream *entering* block i, so blocks 0..start-1 are clean
    for layer in range(start + 1):
        assert np.allclose(base[layer], abl[layer]), layer
    assert not np.allclose(base[start + 1], abl[start + 1])
    assert not np.allclose(base[-1], abl[-1])

    seen = {}

    def record(act, hook):
        seen[hook.name] = act.detach().clone()

    with m.hooks(fwd_hooks=hooks + [(f"blocks.{start}.hook_mlp_out", record)]):
        m(ids)
    v = seen[f"blocks.{start}.hook_mlp_out"][0]
    assert _torch.allclose(v, v[0:1].expand_as(v)), "frozen block still varies by position"

    # every layer must be covered exactly once per component, baseline included
    names = [n for n, _ in conditions(28, 5, ("attn", "mlp"))]
    assert names[0] == "baseline" and len(names) == 1 + 2 * 6, names
    covered = [i for n, hooks_ in conditions(28, 5, ("mlp",))[1:] for i in range(28) if f"blocks.{i}.hook_mlp_out" in [h for h, _ in hooks_]]
    assert sorted(covered) == list(range(28)), covered


def test_capture_shapes():
    """Hooked value vectors must line up with the layers and tokens they came from."""
    from core import model as model_mod

    m, tok, device = model_mod.load(config.MODEL_NAME, "cpu", dtype=torch.float32)
    _, ids = build_input(tok, "Hi", None)
    a = capture.run(m, ids, device)
    n_layers, t = m.cfg.n_layers, ids.shape[1]
    assert a["hidden_states"].shape == (n_layers + 1, t, m.cfg.d_model)
    assert a["attentions"].shape == (n_layers, m.cfg.n_heads, t, t)
    assert a["values"].shape[:2] == (n_layers, t)
    # attention rows are distributions over the causal prefix
    assert np.allclose(a["attentions"].sum(-1), 1.0, atol=1e-4)
    assert np.allclose(np.triu(a["attentions"], k=1), 0.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
