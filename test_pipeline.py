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
