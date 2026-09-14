"""Hook management, activation/attention capture, and on-disk storage.

Built on TransformerLens `HookedTransformer`, so hooks are addressed by name
(`blocks.{l}.attn.hook_v`) rather than by reaching into module attributes.
No experiment-specific logic here.
"""

import json
from pathlib import Path

import numpy as np
import torch


def resid_hook_name(k: int) -> str:
    """Hook whose output *is* hidden_states[k].

    k=0 is the embedding output (blocks.0.hook_resid_pre); k>=1 is the residual
    stream leaving block k-1. One mapping for the whole codebase, so a probe
    fitted at k, a steering vector built at k and an intervention written at k
    all refer to the same tensor.
    """
    return "blocks.0.hook_resid_pre" if k == 0 else f"blocks.{k - 1}.hook_resid_post"


def _np(t, positions=None, axis=0):
    """(1, ...) torch tensor -> unbatched float32 numpy, optionally subset.

    `positions` is an int or sequence of ints into `axis` (counted after the
    batch dimension is dropped); the axis is kept, so a single position comes
    back as length 1 rather than disappearing. Negative indices wrap.

    The subset is taken on-device and before the float32 cast, which is the
    whole point: at 8B the (T, V) logits alone are ~36 MB a pass and callers
    that only score the next token read one row of them.
    """
    t = t.detach()[0]
    if positions is not None:
        idx = torch.as_tensor(np.atleast_1d(positions), dtype=torch.long, device=t.device)
        t = t.index_select(axis, idx % t.shape[axis])
    return t.to("cpu", torch.float32).numpy()


@torch.no_grad()
def run(
    model,
    input_ids,
    device,
    what=("hidden_states", "attentions", "values"),
    interventions=(),
    layers=None,
    positions=None,
):
    """Forward pass capturing hidden states, attention weights, and per-layer value vectors.

    `what` selects which hook families to register and return (default: all
    three, matching prior behaviour exactly).

    `interventions` is a sequence of `(hook_name, fn)` pairs applied as write
    hooks for the duration of the pass, where `fn(activation, hook)` returns the
    replacement activation. Everything captured downstream reflects them.

    `layers` restricts the hidden-state capture to those k (default: all of
    them). The returned array is stacked in the order given, so it is indexed by
    position in `layers`, not by k -- the caller that asked for a subset is the
    one that knows the mapping. `positions` likewise restricts the position axis
    of everything returned. Both only ever drop work: what comes back is
    bit-identical to slicing the full capture, which is what makes them safe to
    reach for when a sweep would otherwise move gigabytes per forward pass.

    Returns dict of numpy arrays (only the keys named in `what`), with L' and T'
    the number of selected layers and positions (all of them by default):
      hidden_states (L', T', D)  -- L' = len(layers), or L+1 by default: index
                                   0 is the embedding output, then the residual
                                   stream after each layer (resid_post). The
                                   last row is pre-final-norm.
      attentions    (L, H, T', T)
      values        (L, T', D_kv) -- GQA: D_kv = n_kv_heads * d_head
      logits        (T', V)       -- output logits, so an intervention can be
                                   scored on behaviour and not just activations.
    """
    input_ids = input_ids.to(device)
    n_layers = model.cfg.n_layers
    ks = list(range(n_layers + 1)) if layers is None else [int(k) for k in layers]

    wanted = set()
    if "hidden_states" in what:
        wanted |= {resid_hook_name(k) for k in ks}
    if "attentions" in what:
        wanted |= {f"blocks.{i}.attn.hook_pattern" for i in range(n_layers)}
    if "values" in what:
        wanted |= {f"blocks.{i}.attn.hook_v" for i in range(n_layers)}

    with model.hooks(fwd_hooks=list(interventions)):
        logits, cache = model.run_with_cache(input_ids, names_filter=lambda n: n in wanted)

    out = {}
    if "logits" in what:
        out["logits"] = _np(logits, positions)
    if "hidden_states" in what:
        out["hidden_states"] = np.stack([_np(cache[resid_hook_name(k)], positions) for k in ks])
    if "attentions" in what:
        # (head, query_pos, key_pos): the query axis is the one `positions` means.
        out["attentions"] = np.stack(
            [_np(cache[f"blocks.{i}.attn.hook_pattern"], positions, axis=1)
             for i in range(n_layers)]
        )
    if "values" in what:
        # hook_v is (batch, pos, n_kv_heads, d_head); flatten the heads back into
        # the v_proj output layout.
        vs = [_np(cache[f"blocks.{i}.attn.hook_v"], positions) for i in range(n_layers)]
        out["values"] = np.stack([v.reshape(v.shape[0], -1) for v in vs])
    return out


def save(path, arrays: dict, meta: dict, compress=True):
    """Write arrays to `path` (.npz) and `meta` alongside as .meta.json.

    `compress=False` uses `np.savez` (faster, larger) instead of
    `np.savez_compressed`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    (np.savez_compressed if compress else np.savez)(path, **arrays)
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def load(path):
    """Returns (arrays dict, meta dict)."""
    path = Path(path)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    return arrays, meta
