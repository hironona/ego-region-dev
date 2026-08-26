"""Hook management, activation/attention capture, and on-disk storage.

Built on TransformerLens `HookedTransformer`, so hooks are addressed by name
(`blocks.{l}.attn.hook_v`) rather than by reaching into module attributes.
No experiment-specific logic here.
"""

import json
from pathlib import Path

import numpy as np
import torch


def _np(t):
    """(1, ...) torch tensor -> unbatched float32 numpy."""
    return t.detach().to("cpu", torch.float32).numpy()[0]


@torch.no_grad()
def run(model, input_ids, device):
    """Forward pass capturing hidden states, attention weights, and per-layer value vectors.

    Returns dict of numpy arrays:
      hidden_states (L+1, T, D)  -- index 0 is the embedding output, then the
                                    residual stream after each layer (resid_post).
                                    The last row is pre-final-norm.
      attentions    (L, H, T, T)
      values        (L, T, D_kv) -- GQA: D_kv = n_kv_heads * d_head
    """
    input_ids = input_ids.to(device)
    n_layers = model.cfg.n_layers
    wanted = {"blocks.0.hook_resid_pre"} | {
        f"blocks.{i}.{h}"
        for i in range(n_layers)
        for h in ("hook_resid_post", "attn.hook_pattern", "attn.hook_v")
    }
    _, cache = model.run_with_cache(input_ids, names_filter=lambda n: n in wanted)

    resid = [_np(cache["blocks.0.hook_resid_pre"])] + [
        _np(cache[f"blocks.{i}.hook_resid_post"]) for i in range(n_layers)
    ]

    return {
        "hidden_states": np.stack(resid),
        "attentions": np.stack(
            [_np(cache[f"blocks.{i}.attn.hook_pattern"]) for i in range(n_layers)]
        ),
        # hook_v is (batch, pos, n_kv_heads, d_head); flatten the heads back into
        # the v_proj output layout.
        "values": np.stack(
            [
                _np(cache[f"blocks.{i}.attn.hook_v"]).reshape(input_ids.shape[1], -1)
                for i in range(n_layers)
            ]
        ),
    }


def save(path, arrays: dict, meta: dict):
    """Write arrays to `path` (.npz) and `meta` alongside as .meta.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def load(path):
    """Returns (arrays dict, meta dict)."""
    path = Path(path)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    return arrays, meta
