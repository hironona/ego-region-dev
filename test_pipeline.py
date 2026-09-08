"""Self-checks for the bits that would silently produce wrong plots.

    uv run python test_pipeline.py
"""

import numpy as np
import torch
from transformers import AutoTokenizer

from core import capture
from experiments.speaker_probe import config
from experiments.speaker_probe.run_capture import build_input

# Each experiment owns its config; these checks are model-level, so they read
# speaker_probe's copy as the representative one.


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


def test_head_ablation_matches_layer_ablation():
    """Freezing all of a layer's heads (hook_z) must equal freezing its attn_out.

    W_O is linear, so the two edits are the same edit. This is what licenses
    comparing a head-sweep number against a window-sweep number; if it ever
    fails, the two experiments are measuring different interventions.
    """
    from core import model as model_mod
    from experiments.head_ablation_sweep.run_capture import Z_HOOK, conditions, mean_ablate_heads
    from experiments.mean_ablation_probe.run_capture import mean_over_context

    m, tok, device = model_mod.load(config.MODEL_NAME, "cpu", dtype=torch.float32)
    ids = tok(
        "<|im_start|>user\nhello there friend how are you<|im_end|>\n",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]

    layer = 2
    by_head = capture.run(
        m, ids, device, what=("hidden_states",),
        interventions=[(Z_HOOK.format(layer=layer), mean_ablate_heads(range(m.cfg.n_heads)))],
    )["hidden_states"]
    by_layer = capture.run(
        m, ids, device, what=("hidden_states",),
        interventions=[(f"blocks.{layer}.hook_attn_out", mean_over_context)],
    )["hidden_states"]
    base = capture.run(m, ids, device, what=("hidden_states",))["hidden_states"]

    assert np.allclose(by_head, by_layer, atol=1e-2), np.abs(by_head - by_layer).max()
    assert not np.allclose(by_head, base), "ablation was a no-op"

    one_head = capture.run(
        m, ids, device, what=("hidden_states",),
        interventions=[(Z_HOOK.format(layer=layer), mean_ablate_heads([3]))],
    )["hidden_states"]
    assert not np.allclose(one_head, base)
    assert np.abs(one_head - base).max() < np.abs(by_head - base).max()

    names = [n for n, _ in conditions((0, 1, 2, 3, 4), m.cfg.n_heads)]
    assert len(names) == 1 + 5 * m.cfg.n_heads + 5 + 1, len(names)


def test_generation_prompt_is_clean_without_enable_thinking():
    """Counter-intuitive and easy to get backwards: on this template
    `enable_thinking=False` ADDS an empty `<think></think>` block to the
    generation prompt (it suppresses thinking by pre-closing it), and the plain
    default is the think-free one. steering_boundary.prompt_ids relies on that.
    """
    from experiments.steering_boundary.run_capture import prompt_ids

    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    text, ids = prompt_ids(tok, "hi", system="be nice")
    assert text.endswith("<|im_start|>assistant\n"), repr(text[-40:])
    assert "<think>" not in text and ids.shape[0] == 1

    flagged = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True,
        tokenize=False, enable_thinking=False,
    )
    assert "<think>" in flagged, "flag no longer inverts; prompt_ids can pass it again"


def test_trait_prompts_are_held_out_and_content_matched():
    """The vector is elicited on eval questions, so two things must hold.

    Overlap with the scored rows would manufacture a steering effect out of items
    the vector was computed from. And every question must appear under both system
    prompts, or the difference in means picks up wording instead of persona.
    """
    from experiments.steering_boundary import config
    from experiments.steering_boundary.data import contrastive_pairs, eval_questions

    scored = {r["question"] for r in eval_questions(
        config.EVAL_SET, config.N_EVAL, config.DATA_DIR)}
    assert len(scored) == config.N_EVAL, "duplicate questions in the scored rows"

    pairs = contrastive_pairs(
        config.TRAIT, config.EVAL_SET, config.N_VECTOR,
        config.DATA_DIR, config.VECTOR_OFFSET)
    assert len(pairs) == 2 * config.N_VECTOR
    fitted = {q for _sys, q, _l in pairs}
    assert not scored & fitted, sorted(scored & fitted)[:3]

    # each question must appear once as trait and once as neutral, or the
    # difference in means is a difference in questions
    systems = {q: set() for q in fitted}
    for sys_prompt, q, label in pairs:
        systems[q].add(label)
    assert all(v == {0, 1} for v in systems.values())


def test_steer_positions_touch_exactly_the_intended_tokens():
    """"all" must shift every position by v; "last" must shift only the final one.

    Getting this wrong is invisible in the output: both modes still produce a
    plausible steering curve, but the crossing fraction would be counted over
    activations that were never pushed.
    """
    from core import model as model_mod
    from experiments.steering_boundary.run_steer import add_vector, hook_name

    m, tok, device = model_mod.load(config.MODEL_NAME, "cpu", dtype=torch.float32)
    ids = tok("<|im_start|>user\nhello there friend<|im_end|>\n",
              add_special_tokens=False, return_tensors="pt")["input_ids"]

    k = 5
    v = torch.zeros(m.cfg.d_model)
    v[7] = 3.0
    base = capture.run(m, ids, device, what=("hidden_states",))["hidden_states"]

    for mode in ("all", "last"):
        out = capture.run(
            m, ids, device, what=("hidden_states",),
            interventions=[(hook_name(k), add_vector(v, mode))],
        )["hidden_states"]
        delta = out[k] - base[k]
        touched = np.abs(delta).max(axis=1) > 1e-4
        expected = np.ones(ids.shape[1], bool) if mode == "all" else (
            np.arange(ids.shape[1]) == ids.shape[1] - 1)
        assert np.array_equal(touched, expected), (mode, touched)
        assert np.allclose(delta[touched], v.numpy(), atol=1e-4), mode
        assert np.allclose(base[k - 1], out[k - 1], atol=1e-4), mode
        assert not np.allclose(base[-1], out[-1]), mode

    # run_steer scores behaviour, so logits must come out of the same call
    logits = capture.run(m, ids, device, what=("logits",))["logits"]
    assert logits.shape == (ids.shape[1], m.cfg.d_vocab)


def test_alpha_star_actually_lands_on_the_boundary():
    """The raw-space undo of the scaler and the crossing formula must agree.

    If either is wrong, alpha* is computed in standardised units and applied in
    residual-stream units, and the marker on the steering plot is meaningless.
    """
    from experiments.steering_boundary.analyze import crossings, fit_speaker_boundary

    rng = np.random.default_rng(0)
    n, d = 400, 12
    conv = np.repeat(np.arange(40), 10)
    y = (rng.random(n) < 0.5).astype(np.uint8)
    # one informative direction, plus offset/scale so the scaler has work to do
    X = rng.normal(size=(n, 1, d)) * np.linspace(0.5, 5, d) + 20.0
    X[:, 0, 3] += 12.0 * y

    W, B, acc = fit_speaker_boundary(X, y, conv, seed=0)
    assert acc[0] > 0.9, acc

    eval_X = rng.normal(size=(25, 1, d)) * 3 + 20.0
    V = rng.normal(size=(1, d))
    alpha, z0, denom = crossings(eval_X, W, B, V)

    # z is linear in alpha, so the steered decision value must be exactly zero
    steered = (eval_X[:, 0, :] + alpha[0][:, None] * V[0]) @ W[0] + B[0]
    assert np.abs(steered).max() < 1e-6, np.abs(steered).max()
    # and z0 must be the probe's own decision value, not a rescaled cousin
    assert np.allclose(z0[0], eval_X[:, 0, :] @ W[0] + B[0])


def test_capture_shapes():
    """Hooked value vectors must line up with the layers and tokens they came from."""
    from core import model as model_mod

    m, tok, device = model_mod.load(config.MODEL_NAME, "cpu", dtype=torch.float32)
    _, ids, _ = build_input(tok, [{"role": "user", "content": "Hi"}])
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
