"""capture.npz -> PCA plots. Never touches the model, so plots regenerate cheaply.

Usage:  uv run python -m experiments.self_user_geometry.analyze
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from core import capture  # noqa: E402

from . import config  # noqa: E402

ROLE_COLORS = {"user": "tab:blue", "assistant": "tab:red", "system": "tab:green"}


def normalize_per_layer(x):
    """(L, T, D) -> per-layer mean-centred, unit-RMS.

    Residual-stream norm grows by an order of magnitude across layers; without this
    PC1 is just "which layer is this" and the role geometry is invisible.
    """
    x = x - x.mean(axis=1, keepdims=True)
    return x / (np.sqrt((x**2).sum(-1, keepdims=True).mean(axis=1, keepdims=True)) + 1e-9)


def _fit(x, n_components):
    """PCA fitted on (N, D)."""
    return PCA(n_components=n_components).fit(x)


def _role_items(meta):
    """[(label, role, position)] for every tracked role token."""
    items = []
    for role, positions in meta["role_positions"].items():
        for k, pos in enumerate(positions):
            label = role if len(positions) == 1 else f"{role}[{k}]"
            items.append((label, role, pos))
    return items


def plot_layer_trajectories(arrays, meta, out_dir, n_components=3):
    """Per-layer trajectory of the role-token hidden states in a shared PCA space."""
    hs = normalize_per_layer(arrays["hidden_states"])  # (L+1, T, D)
    n_layers, _, d = hs.shape
    pca = _fit(hs.reshape(-1, d), min(n_components, d))
    var = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(13, 5.5))
    ax2 = fig.add_subplot(1, 2, 1)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")

    for label, role, pos in _role_items(meta):
        traj = pca.transform(hs[:, pos, :])  # (L+1, k)
        color = ROLE_COLORS.get(role, "tab:gray")
        ax2.plot(traj[:, 0], traj[:, 1], "-o", ms=3, color=color, label=label, alpha=0.85)
        ax2.annotate("L0", traj[0, :2], fontsize=8, color=color)
        ax2.annotate(f"L{n_layers - 1}", traj[-1, :2], fontsize=8, color=color)
        ax3.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-o", ms=3, color=color, label=label)

    ax2.set(
        xlabel=f"PC1 ({var[0]:.1%})",
        ylabel=f"PC2 ({var[1]:.1%})",
        title="Role-token trajectory across layers (2D)",
    )
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax3.set(xlabel="PC1", ylabel="PC2", zlabel=f"PC3 ({var[2]:.1%})", title="(3D)")
    fig.suptitle(f"Hidden-state PCA — {meta['model']}")
    fig.tight_layout()
    _save(fig, out_dir / "layer_trajectories.png")


def plot_all_tokens(arrays, meta, out_dir, layers=(0, None, -1)):
    """All token activations at a few layers, role tokens highlighted."""
    hs = normalize_per_layer(arrays["hidden_states"])
    n_layers, _, d = hs.shape
    pca = _fit(hs.reshape(-1, d), 2)
    picks = [n_layers // 2 if lay is None else lay % n_layers for lay in layers]
    role_pos = {pos: (label, role) for label, role, pos in _role_items(meta)}

    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4.5))
    for ax, lay in zip(np.atleast_1d(axes), picks):
        pts = pca.transform(hs[lay])
        ax.scatter(pts[:, 0], pts[:, 1], s=14, c="lightgray", zorder=1)
        for pos, (label, role) in role_pos.items():
            ax.scatter(
                *pts[pos], s=70, color=ROLE_COLORS.get(role, "k"), zorder=3, label=label
            )
        ax.set(title=f"layer {lay}", xlabel="PC1", ylabel="PC2")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Token activations in shared hidden-state PCA space")
    fig.tight_layout()
    _save(fig, out_dir / "tokens_by_layer.png")


def plot_value_directions(arrays, meta, out_dir):
    """Value-vector directions at the role tokens, in a PCA space fit on all value vectors."""
    vals = normalize_per_layer(arrays["values"])  # (L, T, D_kv)
    n_layers, _, d = vals.shape
    pca = _fit(vals.reshape(-1, d), 2)
    var = pca.explained_variance_ratio_
    items = _role_items(meta)

    fig, (ax_dir, ax_cos) = plt.subplots(1, 2, figsize=(13, 5.5))
    for label, role, pos in items:
        v = pca.transform(vals[:, pos, :])  # (L, 2)
        color = ROLE_COLORS.get(role, "tab:gray")
        # one arrow per layer from the origin; no polyline — layer order is not a path
        for lay in range(n_layers):
            ax_dir.annotate(
                "",
                xy=v[lay],
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    alpha=0.25 + 0.6 * lay / max(n_layers - 1, 1),
                ),
            )
        ax_dir.scatter([], [], color=color, label=label)
        for lay in (0, n_layers - 1):
            ax_dir.annotate(f"L{lay}", v[lay], fontsize=7, color=color)
    ax_dir.axhline(0, color="k", lw=0.5)
    ax_dir.axvline(0, color="k", lw=0.5)
    ax_dir.set(
        xlabel=f"PC1 ({var[0]:.1%})",
        ylabel=f"PC2 ({var[1]:.1%})",
        title="Value-vector directions at role tokens (opacity = depth)",
    )
    ax_dir.legend(fontsize=8)
    ax_dir.grid(alpha=0.3)

    # cosine similarity between each pair of role-token value vectors, per layer
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a = vals[:, items[i][2], :]
            b = vals[:, items[j][2], :]
            cos = (a * b).sum(1) / (
                np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
            )
            ax_cos.plot(cos, "-o", ms=3, label=f"{items[i][0]} · {items[j][0]}")
    ax_cos.axhline(0, color="k", lw=0.5)
    ax_cos.set(
        xlabel="layer",
        ylabel="cosine similarity",
        title="Role-token value-vector alignment per layer",
    )
    ax_cos.legend(fontsize=8)
    ax_cos.grid(alpha=0.3)
    fig.suptitle(f"Value-vector geometry — {meta['model']}")
    fig.tight_layout()
    _save(fig, out_dir / "value_directions.png")


def plot_role_attention(arrays, meta, out_dir):
    """How much each role token is attended to, averaged over heads, per layer."""
    attn = arrays["attentions"]  # (L, H, T, T)
    received = attn.mean(1).sum(1)  # (L, T) total incoming attention per key position
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, role, pos in _role_items(meta):
        ax.plot(
            received[:, pos], "-o", ms=3, color=ROLE_COLORS.get(role, "k"), label=label
        )
    ax.set(
        xlabel="layer",
        ylabel="summed attention received (head-mean)",
        title="Attention received by role tokens",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "role_attention.png")


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", default=str(config.CAPTURE_PATH))
    p.add_argument("--out", default=str(config.PLOT_DIR))
    args = p.parse_args()

    arrays, meta = capture.load(args.capture)
    out_dir = Path(args.out)
    plot_layer_trajectories(arrays, meta, out_dir)
    plot_all_tokens(arrays, meta, out_dir)
    plot_value_directions(arrays, meta, out_dir)
    plot_role_attention(arrays, meta, out_dir)


if __name__ == "__main__":
    main()
