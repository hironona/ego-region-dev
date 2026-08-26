"""captures/*.npz -> turn-agnostic probe accuracy per (condition, layer) + plots.

Never touches the model.

Usage:  uv run python -m experiments.mean_ablation_probe.analyze
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

# The probe and the conversation-grouped split are the speaker-probe experiment's,
# so a number here is comparable with a number there.
from experiments.speaker_probe.analyze import conv_train_test_masks, probe_accuracy  # noqa: E402

from . import config  # noqa: E402


def condition_sort_key(name):
    """baseline first, then attn windows in layer order, then mlp windows."""
    if name == "baseline":
        return (0, "", 0)
    comp, span = name.split("_")
    return (1, comp, int(span.split("-")[0]))


def accuracies(capture_dir, seed):
    """{condition: [accuracy per layer]} over every capture in the directory."""
    paths = sorted(Path(capture_dir).glob("*.npz"), key=lambda p: condition_sort_key(p.stem))
    if not paths:
        raise FileNotFoundError(f"no captures in {capture_dir} — run run_capture first")

    out, meta_of = {}, {}
    for path in paths:
        arrays, meta = capture.load(path)
        X = arrays["X"].astype(np.float32)
        speaker, conv = arrays["speaker"], arrays["conv"]
        train_mask, test_mask = conv_train_test_masks(conv, seed=seed)
        out[path.stem] = [
            probe_accuracy(X[:, layer, :], speaker, train_mask, test_mask)
            for layer in range(X.shape[1])
        ]
        meta_of[path.stem] = meta
        print(f"probed {path.stem}")
    return out, meta_of


def ablated_span(name):
    """(start, end_inclusive) of the frozen window, or None for the baseline."""
    m = re.match(r"(attn|mlp)_(\d+)-(\d+)$", name)
    return None if m is None else (int(m.group(2)), int(m.group(3)))


def plot_lines(acc, out_dir, components=("attn", "mlp")):
    """Accuracy vs layer, one line per frozen window, one panel per component."""
    fig, axes = plt.subplots(1, len(components), figsize=(6.5 * len(components), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    base = acc["baseline"]
    layers = np.arange(len(base))

    for ax, comp in zip(axes, components):
        names = [n for n in acc if n.startswith(f"{comp}_")]
        names.sort(key=condition_sort_key)
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(names)))
        ax.plot(layers, base, "--", color="k", lw=1.5, label="baseline (no ablation)")
        for name, color in zip(names, colors):
            start, end = ablated_span(name)
            ax.plot(layers, acc[name], "-o", ms=2.5, color=color, label=f"froze {start}-{end}")
            # mark which layers this condition actually silenced
            ax.axvspan(start, end + 1, color=color, alpha=0.06)
        ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        ax.set(xlabel="layer the probe reads", title=f"{comp}_out mean-ablated")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("balanced accuracy")
    fig.suptitle("Speaker separability after freezing a window of blocks", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_dir / "accuracy_by_layer.png")


def plot_delta_heatmaps(acc, out_dir, components=("attn", "mlp")):
    """accuracy(condition) - accuracy(baseline), the plot that localises the source."""
    base = np.array(acc["baseline"])
    fig, axes = plt.subplots(1, len(components), figsize=(6.8 * len(components), 4.6))
    axes = np.atleast_1d(axes)
    lim = max(
        abs(np.array(acc[n]) - base).max()
        for n in acc
        if n != "baseline" and np.isfinite(acc[n]).any()
    )

    for ax, comp in zip(axes, components):
        names = [n for n in acc if n.startswith(f"{comp}_")]
        names.sort(key=condition_sort_key)
        delta = np.array([np.array(acc[n]) - base for n in names])
        im = ax.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        for row, name in enumerate(names):
            start, end = ablated_span(name)
            # outline the frozen window: inside it the drop is direct, outside it downstream
            ax.add_patch(
                plt.Rectangle(
                    (start - 0.5, row - 0.5), end - start + 2, 1,
                    fill=False, edgecolor="k", lw=1.2,
                )
            )
        ax.set(
            xlabel="layer the probe reads",
            ylabel="frozen window",
            title=f"{comp}_out — accuracy minus baseline",
            yticks=range(len(names)),
            yticklabels=[f"{a}-{b}" for a, b in (ablated_span(n) for n in names)],
        )
        fig.colorbar(im, ax=ax, label="Δ balanced accuracy")
    fig.suptitle("Where the speaker signal is written (black box = frozen blocks)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_dir / "delta_heatmap.png")


def plot_final_layer(acc, out_dir):
    """One bar per condition: what survives to the last layer."""
    names = sorted(acc, key=condition_sort_key)
    vals = [acc[n][-1] for n in names]
    colors = ["k" if n == "baseline" else ("tab:orange" if n.startswith("attn") else "tab:purple") for n in names]
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(names)), 4.4))
    ax.bar(range(len(names)), vals, color=colors)
    ax.axhline(acc["baseline"][-1], color="k", lw=0.9, ls="--", label="baseline")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":", label="chance")
    ax.set(
        ylabel="balanced accuracy at final layer",
        title="Speaker separability at the last layer, per frozen window",
        xticks=range(len(names)),
        ylim=(0.4, 1.02),
    )
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, out_dir / "final_layer_bars.png")


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

    acc, meta_of = accuracies(args.capture_dir, seed=config.SEED)
    out_dir = Path(args.out_dir)
    components = tuple(dict.fromkeys(n.split("_")[0] for n in acc if n != "baseline"))

    plot_lines(acc, out_dir, components)
    plot_delta_heatmaps(acc, out_dir, components)
    plot_final_layer(acc, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "model": meta_of["baseline"]["model"],
                "ablation": "mean over context positions",
                "window": meta_of["baseline"]["window"],
                "note": "balanced accuracy, turn-agnostic, split by conversation",
                "accuracy_by_condition": acc,
            },
            indent=2,
        )
    )
    print(f"wrote {out_dir / 'results.json'}")

    base = np.array(acc["baseline"])
    print(f"\n{'condition':>14}  final-layer  min-over-layers  largest drop vs baseline")
    for name in sorted(acc, key=condition_sort_key):
        a = np.array(acc[name])
        drop = (base - a).max()
        print(f"{name:>14}  {a[-1]:11.3f}  {np.nanmin(a):15.3f}  {drop:24.3f}")


if __name__ == "__main__":
    main()
