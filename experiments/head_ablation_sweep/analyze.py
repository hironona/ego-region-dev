"""captures/*.npz -> per-head probe accuracy + plots. Never touches the model.

Usage:  uv run python -m experiments.head_ablation_sweep.analyze
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core import capture  # noqa: E402
from experiments.speaker_probe.analyze import (  # noqa: E402
    conv_train_test_masks,
    probe_accuracy,
)

from . import config  # noqa: E402

HEAD_RE = re.compile(r"^L(\d+)H(\d+)$")


def sort_key(name):
    """baseline, then heads in (layer, head) order, then whole layers, then window."""
    m = HEAD_RE.match(name)
    if name == "baseline":
        return (0, 0, 0)
    if m:
        return (1, int(m.group(1)), int(m.group(2)))
    if name.endswith("all"):
        return (2, int(name[1:3]), 0)
    return (3, 0, 0)


def accuracies(capture_dir, seed):
    """{condition: {probe_layer: accuracy}} plus the metadata of the baseline."""
    paths = sorted(Path(capture_dir).glob("*.npz"), key=lambda p: sort_key(p.stem))
    if not paths:
        raise FileNotFoundError(f"no captures in {capture_dir} — run run_capture first")

    out, base_meta = {}, None
    for path in paths:
        arrays, meta = capture.load(path)
        X = arrays["X"].astype(np.float32)
        train_mask, test_mask = conv_train_test_masks(arrays["conv"], seed=seed)
        out[path.stem] = {
            int(layer): probe_accuracy(X[:, i, :], arrays["speaker"], train_mask, test_mask)
            for i, layer in enumerate(meta["probe_layers"])
        }
        if path.stem == "baseline":
            base_meta = meta
        print(f"probed {path.stem}")
    if base_meta is None:
        raise FileNotFoundError("no baseline.npz in the capture directory")
    return out, base_meta


def head_grid(acc, sweep_layers, n_heads, probe_layer):
    """(n_sweep_layers, n_heads) accuracies at one readout layer; NaN where missing."""
    grid = np.full((len(sweep_layers), n_heads), np.nan)
    for name, per_layer in acc.items():
        m = HEAD_RE.match(name)
        if m and int(m.group(1)) in sweep_layers:
            grid[sweep_layers.index(int(m.group(1))), int(m.group(2))] = per_layer[probe_layer]
    return grid


def plot_head_heatmaps(acc, sweep_layers, n_heads, probe_layers, out_dir):
    """Δ accuracy vs baseline for every (layer, head), one panel per readout layer."""
    fig, axes = plt.subplots(1, len(probe_layers), figsize=(5.4 * len(probe_layers), 3.6))
    axes = np.atleast_1d(axes)
    grids = [head_grid(acc, sweep_layers, n_heads, pl) - acc["baseline"][pl] for pl in probe_layers]
    lim = max(0.02, np.nanmax(np.abs(np.array(grids))))

    for ax, pl, grid in zip(axes, probe_layers, grids):
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set(
            xlabel="head",
            ylabel="layer",
            title=f"probe reads layer {pl}",
            xticks=range(n_heads),
            yticks=range(len(sweep_layers)),
            yticklabels=sweep_layers,
        )
        ax.set_xticklabels(range(n_heads), fontsize=7)
        fig.colorbar(im, ax=ax, label="Δ balanced accuracy")
    fig.suptitle("Single-head mean ablation — accuracy minus baseline", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_dir / "head_delta_heatmap.png")


def plot_ranked_heads(acc, probe_layer, out_dir, top=20):
    """The heads that matter, ranked, against the layer and window references."""
    base = acc["baseline"][probe_layer]
    heads = {n: v[probe_layer] for n, v in acc.items() if HEAD_RE.match(n)}
    ranked = sorted(heads.items(), key=lambda kv: kv[1])[:top]

    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(ranked)), 4.6))
    ax.bar(range(len(ranked)), [v for _, v in ranked], color="tab:red")
    ax.axhline(base, color="k", ls="--", lw=1.0, label=f"baseline ({base:.3f})")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8, label="chance")
    window = next((n for n in acc if n.startswith("window_")), None)
    if window:
        ax.axhline(
            acc[window][probe_layer],
            color="tab:blue",
            ls="-.",
            lw=1.0,
            label=f"whole window ({acc[window][probe_layer]:.3f})",
        )
    ax.set(
        ylabel="balanced accuracy",
        title=f"Most damaging single heads (probe reads layer {probe_layer})",
        xticks=range(len(ranked)),
        ylim=(0.4, 1.02),
    )
    ax.set_xticklabels([n for n, _ in ranked], rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, out_dir / "ranked_heads.png")


def plot_additivity(acc, sweep_layers, n_heads, probe_layer, out_dir):
    """Do single-head drops account for the whole-layer drop, or is it distributed?

    x: summed drop of a layer's 16 individual heads. y: drop when that whole layer
    is frozen at once. Far below the diagonal means the heads are redundant — each
    one alone is dispensable because the others still carry the signal.
    """
    base = acc["baseline"][probe_layer]
    xs, ys, labels = [], [], []
    for layer in sweep_layers:
        singles = [
            base - acc[f"L{layer:02d}H{h:02d}"][probe_layer]
            for h in range(n_heads)
            if f"L{layer:02d}H{h:02d}" in acc
        ]
        whole = f"L{layer:02d}all"
        if not singles or whole not in acc:
            continue
        xs.append(sum(singles))
        ys.append(base - acc[whole][probe_layer])
        labels.append(f"L{layer}")

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    top = max(xs + ys + [0.01]) * 1.15
    ax.plot([0, top], [0, top], "--", color="grey", lw=0.9, label="additive")
    ax.scatter(xs, ys, s=45, color="tab:purple", zorder=3)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set(
        xlabel="sum of individual head drops",
        ylabel="drop when the whole layer is frozen",
        title=f"Additivity of head effects (layer {probe_layer} readout)",
        xlim=(0, top),
        ylim=(0, top),
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "additivity.png")


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture-dir", default=str(config.CAPTURE_DIR))
    p.add_argument("--out-dir", default=str(config.PLOT_DIR))
    args = p.parse_args()

    acc, base_meta = accuracies(args.capture_dir, seed=config.SEED)
    probe_layers = [int(x) for x in base_meta["probe_layers"]]
    n_heads = int(base_meta["n_heads"])
    sweep_layers = sorted({int(HEAD_RE.match(n).group(1)) for n in acc if HEAD_RE.match(n)})
    final = probe_layers[-1]
    out_dir = Path(args.out_dir)

    plot_head_heatmaps(acc, sweep_layers, n_heads, probe_layers, out_dir)
    plot_ranked_heads(acc, final, out_dir)
    plot_additivity(acc, sweep_layers, n_heads, final, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "model": base_meta["model"],
                "ablation": base_meta["ablation"],
                "probe_layers": probe_layers,
                "n_conversations": base_meta["n_conversations"],
                "note": "balanced accuracy, turn-agnostic, split by conversation",
                "accuracy_by_condition": {k: {str(i): v for i, v in d.items()} for k, d in acc.items()},
            },
            indent=2,
        )
    )
    print(f"wrote {out_dir / 'results.json'}")

    base = acc["baseline"][final]
    print(f"\nbaseline at layer {final}: {base:.3f}")
    print(f"{'condition':>12}  {'acc':>6}  {'drop':>6}")
    for name in sorted(acc, key=lambda n: acc[n][final]):
        if name == "baseline":
            continue
        print(f"{name:>12}  {acc[name][final]:6.3f}  {base - acc[name][final]:6.3f}")


if __name__ == "__main__":
    main()
