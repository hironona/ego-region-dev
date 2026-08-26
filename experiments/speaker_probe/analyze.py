"""capture.npz -> per-layer speaker-probe accuracy + plots. Never touches the model.

Usage:  uv run python -m experiments.speaker_probe.analyze
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from core import capture  # noqa: E402

from . import config  # noqa: E402

TURN_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def conv_train_test_masks(conv, test_size=0.25, seed=0):
    """Row masks (train, test) split by conversation id, not by row."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(conv))
    train_idx, test_idx = next(splitter.split(idx, groups=conv))
    train_mask = np.zeros(len(conv), dtype=bool)
    test_mask = np.zeros(len(conv), dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    return train_mask, test_mask


def probe_accuracy(X_layer, y, train_mask, test_mask):
    """Balanced accuracy of a StandardScaler+LogisticRegression probe on one layer.

    Returns NaN (rather than raising) if either split is degenerate — a single
    class in train or test, which a tiny/turn-subset split can produce.
    """
    y_train, y_test = y[train_mask], y[test_mask]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(X_layer[train_mask], y_train)
    preds = clf.predict(X_layer[test_mask])
    # balanced accuracy: classes are balanced by construction (equal user/assistant
    # sampling), but average per-class recall anyway to be explicit about it.
    accs = [
        (preds[y_test == c] == c).mean() if (y_test == c).any() else float("nan")
        for c in (0, 1)
    ]
    return float(np.nanmean(accs))


def exp1_turn_agnostic(X, speaker, train_mask, test_mask):
    """One probe per layer on all rows. Returns list of accuracy, len L+1."""
    n_layers = X.shape[1]
    return [
        probe_accuracy(X[:, layer, :], speaker, train_mask, test_mask)
        for layer in range(n_layers)
    ]


def exp2_turn_sensitive(X, speaker, turn, conv, train_mask, test_mask):
    """Per layer, per turn-pair-subset accuracy. Returns (L+1, 5) array."""
    n_layers = X.shape[1]
    results = np.full((n_layers, len(TURN_PAIRS)), np.nan)
    for j, pair in enumerate(TURN_PAIRS):
        subset = np.isin(turn, pair)
        sub_train = train_mask & subset
        sub_test = test_mask & subset
        for layer in range(n_layers):
            results[layer, j] = probe_accuracy(
                X[:, layer, :], speaker, sub_train, sub_test
            )
    return results


def plot_exp1(acc, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    layers = np.arange(len(acc))
    ax.plot(layers, acc, "-o", ms=3, color="tab:blue")
    ax.axhline(0.5, color="k", lw=0.8, ls="--", label="chance")
    ax.set(
        xlabel="layer",
        ylabel="balanced accuracy",
        title="Exp 1 — turn-agnostic speaker probe",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "exp1_accuracy_by_layer.png")


def plot_exp2_lines(acc2d, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    layers = np.arange(acc2d.shape[0])
    for j, pair in enumerate(TURN_PAIRS):
        ax.plot(layers, acc2d[:, j], "-o", ms=3, label=f"turns {pair[0]},{pair[1]}")
    ax.axhline(0.5, color="k", lw=0.8, ls="--", label="chance")
    ax.set(
        xlabel="layer",
        ylabel="balanced accuracy",
        title="Exp 2 — per-turn-pair speaker probe",
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "exp2_accuracy_by_layer_turnset.png")


def plot_exp2_heatmap(acc2d, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(acc2d, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set(
        xlabel="turn-pair set",
        ylabel="layer",
        title="Exp 2 — accuracy heatmap",
        xticks=range(len(TURN_PAIRS)),
        xticklabels=[f"{a},{b}" for a, b in TURN_PAIRS],
    )
    fig.colorbar(im, ax=ax, label="balanced accuracy")
    fig.tight_layout()
    _save(fig, out_dir / "exp2_heatmap.png")


def plot_pca_boundaries(X, speaker, train_mask, test_mask, out_dir, seed=0, max_points=1500):
    """Per layer: 2D PCA of the activations + the decision boundary of a probe
    fitted in that 2D space, points coloured by speaker.

    The boundary is a probe refitted on the two principal components, not the
    full-D probe from Exp 1 — a hyperplane in 1024-D has no faithful 2D image.
    Each panel therefore reports its own 2D balanced accuracy so the gap against
    the full-D number is visible rather than implied.
    """
    n_layers = X.shape[1]
    ncols = 6
    nrows = int(np.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows))
    rng = np.random.default_rng(seed)
    acc2d = []

    for layer in range(n_layers):
        ax = axes.flat[layer]
        reducer = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=seed))
        z_train = reducer.fit_transform(X[train_mask, layer, :])
        z_test = reducer.transform(X[test_mask, layer, :])
        y_train, y_test = speaker[train_mask], speaker[test_mask]

        clf = LogisticRegression(max_iter=1000).fit(z_train, y_train)
        preds = clf.predict(z_test)
        acc = float(np.mean([(preds[y_test == c] == c).mean() for c in (0, 1)]))
        acc2d.append(acc)

        # shade the two half-planes, then overlay the boundary itself
        pad = 0.05 * np.ptp(z_test, axis=0)
        (x0, y0), (x1, y1) = z_test.min(0) - pad, z_test.max(0) + pad
        gx, gy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
        grid = clf.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        ax.contourf(gx, gy, grid > 0, levels=[-0.5, 0.5, 1.5], colors=["#c6dbef", "#fcbba1"], alpha=0.55)
        ax.contour(gx, gy, grid, levels=[0], colors="k", linewidths=1.0)

        keep = rng.permutation(len(z_test))[:max_points]
        for c, color, label in ((0, "tab:blue", "user"), (1, "tab:red", "assistant")):
            sel = keep[y_test[keep] == c]
            ax.scatter(z_test[sel, 0], z_test[sel, 1], s=3, c=color, alpha=0.5, lw=0, label=label)

        var = reducer[-1].explained_variance_ratio_
        ax.set_title(f"L{layer}  2D acc {acc:.2f}  var {var.sum():.0%}", fontsize=8)
        ax.set(xlim=(x0, x1), ylim=(y0, y1), xticks=[], yticks=[])

    for ax in axes.flat[n_layers:]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", markerscale=4, fontsize=10)
    fig.suptitle("Speaker probe in 2D PCA space (held-out conversations)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, out_dir / "pca_decision_boundaries.png")
    return acc2d


def plot_probe_plane_boundaries(X, speaker, train_mask, test_mask, out_dir, seed=0, max_points=1500):
    """Same idea, but the 2D basis is (probe direction, leading PC orthogonal to it).

    Pure PCA hides the speaker split because the probe direction carries almost
    no variance, so its panels look like noise. Swapping PC1 for the probe's own
    weight vector keeps the plane 2D and honest — the vertical line is the *real*
    full-D boundary, not a refit — and PC1-minus-probe stays on the y axis to show
    what the dominant variance direction is doing meanwhile.
    """
    n_layers = X.shape[1]
    ncols = 6
    nrows = int(np.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows))
    rng = np.random.default_rng(seed)

    for layer in range(n_layers):
        ax = axes.flat[layer]
        scaler = StandardScaler().fit(X[train_mask, layer, :])
        z_train, z_test = scaler.transform(X[train_mask, layer, :]), scaler.transform(X[test_mask, layer, :])
        y_train, y_test = speaker[train_mask], speaker[test_mask]

        clf = LogisticRegression(max_iter=1000).fit(z_train, y_train)
        w = clf.coef_[0]
        w_hat = w / np.linalg.norm(w)

        pc1 = PCA(n_components=1, random_state=seed).fit(z_train).components_[0]
        perp = pc1 - (pc1 @ w_hat) * w_hat  # PC1 with the probe direction removed
        perp /= np.linalg.norm(perp) + 1e-12

        u = z_test @ w_hat
        v = z_test @ perp
        boundary = -clf.intercept_[0] / np.linalg.norm(w)  # where w·z + b = 0

        keep = rng.permutation(len(u))[:max_points]
        pad = 0.05 * np.ptp(u)
        ax.axvspan(u.min() - pad, boundary, color="#c6dbef", alpha=0.55)
        ax.axvspan(boundary, u.max() + pad, color="#fcbba1", alpha=0.55)
        ax.axvline(boundary, color="k", lw=1.0)
        for c, color, label in ((0, "tab:blue", "user"), (1, "tab:red", "assistant")):
            sel = keep[y_test[keep] == c]
            ax.scatter(u[sel], v[sel], s=3, c=color, alpha=0.5, lw=0, label=label)

        ax.set_title(f"L{layer}", fontsize=8)
        ax.set(xlim=(u.min() - pad, u.max() + pad), xticks=[], yticks=[])

    for ax in axes.flat[n_layers:]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", markerscale=4, fontsize=10)
    fig.suptitle(
        "Full-D probe boundary — x: probe direction, y: leading PC orthogonal to it",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, out_dir / "probe_plane_boundaries.png")


def plot_pca_vs_full(acc_full, acc_2d, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    layers = np.arange(len(acc_full))
    ax.plot(layers, acc_full, "-o", ms=3, label="full 1024-D probe")
    ax.plot(layers, acc_2d, "-s", ms=3, label="probe on 2 PCs")
    ax.axhline(0.5, color="k", lw=0.8, ls="--", label="chance")
    ax.set(xlabel="layer", ylabel="balanced accuracy", title="How much survives the 2D projection")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "pca_2d_vs_full.png")


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", default=str(config.CAPTURE_PATH))
    p.add_argument("--out-dir", default=str(config.PLOT_DIR))
    args = p.parse_args()

    arrays, meta = capture.load(args.capture)
    X = arrays["X"].astype(np.float32)  # stored fp16
    speaker = arrays["speaker"]
    turn = arrays["turn"]
    conv = arrays["conv"]

    out_dir = Path(args.out_dir)
    train_mask, test_mask = conv_train_test_masks(conv, seed=config.SEED)

    exp1_acc = exp1_turn_agnostic(X, speaker, train_mask, test_mask)
    exp2_acc = exp2_turn_sensitive(X, speaker, turn, conv, train_mask, test_mask)

    plot_exp1(exp1_acc, out_dir)
    plot_exp2_lines(exp2_acc, out_dir)
    plot_exp2_heatmap(exp2_acc, out_dir)
    pca_acc = plot_pca_boundaries(X, speaker, train_mask, test_mask, out_dir, seed=config.SEED)
    plot_probe_plane_boundaries(X, speaker, train_mask, test_mask, out_dir, seed=config.SEED)
    plot_pca_vs_full(exp1_acc, pca_acc, out_dir)

    results = {
        "model": meta.get("model"),
        "note": "balanced accuracy (mean per-class recall); classes balanced by construction",
        "exp1_turn_agnostic": exp1_acc,
        "pca_2d_turn_agnostic": pca_acc,
        "exp2_turn_sensitive": {
            f"{a},{b}": exp2_acc[:, j].tolist() for j, (a, b) in enumerate(TURN_PAIRS)
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out_dir / 'results.json'}")

    print("layer  exp1_acc")
    for layer, acc in enumerate(exp1_acc):
        print(f"{layer:5d}  {acc:.3f}" if acc == acc else f"{layer:5d}    nan")


if __name__ == "__main__":
    main()
