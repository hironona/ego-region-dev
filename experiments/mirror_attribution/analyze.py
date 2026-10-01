"""capture.npz -> self vector (vectors.npz) + its check, then mirror.npz -> plots.
Never loads the model.

Two stages because run_mirror.py sits between them:

    uv run python -m experiments.mirror_attribution.analyze     # vectors.npz, self_vector.png
    uv run python -m experiments.mirror_attribution.run_mirror  # mirror.npz
    uv run python -m experiments.mirror_attribution.analyze     # + mirror plots

The second call re-derives vectors.npz identically (cheap and deterministic) and
additionally plots the sweep, so there is no stage flag to get wrong.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core import capture  # noqa: E402
from experiments.speaker_probe.analyze import conv_train_test_masks  # noqa: E402
from experiments.steering_boundary.analyze import cosines, fit_speaker_boundary  # noqa: E402

from . import config  # noqa: E402


def self_vectors(X, y, train):
    """Per layer: v = mu_asst - mu_user and b = -v.(mu_asst + mu_user)/2.

    b puts the boundary v.x + b = 0 at the midpoint of the two training means,
    so p = sigma(v.x + b) is 0.5 exactly halfway between the speakers. Without
    b the boundary passes through the origin, which nothing about the residual
    stream centres on. One layer at a time: the whole capture in float64 is
    several GB at 8B.
    """
    n_layers, d = X.shape[1], X.shape[2]
    V, B = np.zeros((n_layers, d)), np.zeros(n_layers)
    for k in range(n_layers):
        Xk = X[train, k, :].astype(np.float64)
        mu_a, mu_u = Xk[y[train] == 1].mean(0), Xk[y[train] == 0].mean(0)
        V[k] = mu_a - mu_u
        B[k] = -V[k] @ (mu_a + mu_u) / 2
    return V, B


def balanced_accuracy(pred, y):
    return float(np.mean([(pred[y == c] == c).mean() for c in (0, 1)]))


def self_vector_accuracy(X, y, V, B, rows):
    """Per layer balanced accuracy of p = sigma(v.x + b) > 0.5 on `rows`."""
    return np.array([
        balanced_accuracy((X[rows, k, :].astype(np.float64) @ V[k] + B[k]) > 0, y[rows])
        for k in range(X.shape[1])
    ])


def fit(cap):
    """capture -> vectors dict (V, B, the reference probe, and their checks)."""
    X, y, conv = cap["X"], cap["y"], cap["conv"]
    train, test = conv_train_test_masks(conv, test_size=config.TEST_SIZE, seed=config.SEED)
    V, B = self_vectors(X, y, train)
    acc = self_vector_accuracy(X, y, V, B, test)
    # The formula as first written, p = sigma(v.x): same v, no offset.
    acc_nobias = self_vector_accuracy(X, y, V, np.zeros_like(B), test)
    # The full logistic probe, as the ceiling a single difference of means is
    # compared against. Same GroupShuffleSplit and seed, so the same held-out
    # conversations.
    W, BW, acc_lr = fit_speaker_boundary(X, y, conv, seed=config.SEED, test_size=config.TEST_SIZE)
    # per layer, in float32: squared norms overflow float16
    x_norm = np.array([np.median(np.linalg.norm(X[:, k, :].astype(np.float32), axis=-1))
                       for k in range(X.shape[1])])
    vec = {
        "V": V, "B": B, "W": W, "BW": BW, "acc": acc, "acc_nobias": acc_nobias,
        "acc_lr": acc_lr, "cos": cosines(W, V), "v_norm": np.linalg.norm(V, axis=1),
        "x_norm": x_norm, "test_conv": np.unique(conv[test]),
    }
    return vec


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_self_vector(vec, model, out_dir):
    ks = np.arange(len(vec["acc"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.plot(ks, vec["acc"], "o-", ms=3, label=r"self vector $\sigma(v\cdot x+b)$")
    ax1.plot(ks, vec["acc_nobias"], ":", color="0.5", label=r"$\sigma(v\cdot x)$, no bias")
    ax1.plot(ks, vec["acc_lr"], "--", color="k", lw=1, label="logistic probe (ceiling)")
    ax1.axhline(0.5, color="0.8", lw=0.8)
    ax1.set(xlabel="layer k (hidden_states[k])", ylabel="balanced accuracy (held-out convs)",
            ylim=(0.4, 1.02), title="does one direction separate the speakers?")
    ax1.legend(fontsize=8)
    ax2.plot(ks, vec["cos"], "o-", ms=3, label="cos(v, w_probe)")
    ax2.plot(ks, vec["v_norm"] / vec["x_norm"], "s-", ms=3, label="|v| / median |x|")
    ax2.set(xlabel="layer k", title="geometry")
    ax2.legend(fontsize=8)
    fig.suptitle(model, fontsize=9)
    _save(fig, out_dir / "self_vector.png")


# --- mirror sweep ---------------------------------------------------------

def choices(logp):
    """1 where the answer "Assistant" beats the answer "User"."""
    return (logp[..., 1] > logp[..., 0]).astype(np.uint8)


def cell_metrics(logp, mass, labels, base_choice):
    """Reduce one cell's per-item arrays (last axis = item) to scalars."""
    ch = choices(logp)
    hit = ch == labels
    correct = np.where(labels == 1, logp[..., 1] - logp[..., 0], logp[..., 0] - logp[..., 1])
    return {
        "acc": float(hit.mean()),
        # asked about a user / an assistant statement: the one-role modes should
        # move only one of these
        "acc_user": float(hit[labels == 0].mean()),
        "acc_assistant": float(hit[labels == 1].mean()),
        "flip": float((ch != base_choice).mean()),
        "margin": float(correct.mean()),
        "answer_mass": float(np.median(mass)),
    }


def summarise(mir, meta):
    """mirror.npz -> nested dict of metrics, keyed n -> kind -> mode -> per-layer lists."""
    turns, layers = mir["turns"], mir["layers"]
    out = {}
    for a, n in enumerate(turns):
        labels = mir["labels"][a]
        base_ch = choices(mir["base_logp"][a])
        base = cell_metrics(mir["base_logp"][a], mir["base_mass"][a], labels, base_ch)
        cells = {}
        for t, kind in enumerate(meta["kinds"]):
            cells[kind] = {}
            for c, mode in enumerate(meta["modes"]):
                per = [cell_metrics(mir["logp"][a, t, i, c], mir["mass"][a, t, i, c],
                                    labels, base_ch) for i in range(len(layers))]
                cells[kind][mode] = {key: [p[key] for p in per] for key in per[0]}
                cells[kind][mode]["disp"] = mir["disp"][a, :, c].tolist()
        out[int(n)] = {
            "baseline": base, "cells": cells,
            "z_acc_user": mir["z_acc"][a, :, 0].tolist(),
            "z_acc_assistant": mir["z_acc"][a, :, 1].tolist(),
        }
    return out


def _grid(n_rows, n_cols):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    return fig, axes


def _plot_line(ax, x, y, mass, min_mass, **kw):
    """Line with filled markers where the model still answers and hollow ones
    where it does not (median answer mass below min_mass)."""
    y, ok = np.asarray(y), np.asarray(mass) >= min_mass
    line, = ax.plot(x, y, "-", **kw)
    ax.plot(x[ok], y[ok], "o", ms=4, color=line.get_color())
    ax.plot(x[~ok], y[~ok], "o", ms=4, mfc="none", color=line.get_color())


def plot_mirror(summary, mir, meta, out_dir):
    turns, layers = mir["turns"], mir["layers"]
    modes, kinds = meta["modes"], meta["kinds"]
    min_mass = meta["min_answer_mass"]
    title = f"{meta['model']}  ({meta['n_eval']} items per n; hollow = answer mass < {min_mass})"

    # 1. accuracy: self vs random vs unmirrored
    fig, axes = _grid(len(modes), len(turns))
    for a, n in enumerate(turns):
        s = summary[int(n)]
        for c, mode in enumerate(modes):
            ax = axes[c, a]
            ax.axhline(0.5, color="0.85", lw=0.8)
            ax.axhline(s["baseline"]["acc"], color="k", ls=":", lw=1, label="unmirrored")
            if mode == "history":
                ax.axhline(1 - s["baseline"]["acc"], color="C3", ls=":", lw=1,
                           label="full swap (1 - unmirrored)")
            for kind, style in (("self", {"color": "C0"}), ("random", {"color": "0.55", "ls": "--"})):
                if kind in kinds:
                    cell = s["cells"][kind][mode]
                    _plot_line(ax, layers, cell["acc"], cell["answer_mass"], min_mass,
                               label=f"mirror ({kind})", **style)
            ax.set_ylim(-0.03, 1.03)
            if c == 0:
                ax.set_title(f"n = {n}")
            if a == 0:
                ax.set_ylabel(f"{mode}\naccuracy")
            if c == len(modes) - 1:
                ax.set_xlabel("mirrored layer k")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title, fontsize=9)
    _save(fig, out_dir / "mirror_accuracy.png")

    # 2. accuracy split by who actually said the quoted statement (self only)
    fig, axes = _grid(len(modes), len(turns))
    for a, n in enumerate(turns):
        s = summary[int(n)]
        for c, mode in enumerate(modes):
            ax = axes[c, a]
            cell = s["cells"]["self"][mode]
            for key, color in (("acc_user", "C2"), ("acc_assistant", "C1")):
                ax.axhline(s["baseline"][key], color=color, ls=":", lw=1)
                _plot_line(ax, layers, cell[key], cell["answer_mass"], min_mass, color=color,
                           label=f"{key.split('_')[1]} said it")
            ax.set_ylim(-0.03, 1.03)
            if c == 0:
                ax.set_title(f"n = {n}")
            if a == 0:
                ax.set_ylabel(f"{mode}\naccuracy")
            if c == len(modes) - 1:
                ax.set_xlabel("mirrored layer k")
    axes[0, 0].legend(fontsize=7, title="dotted = unmirrored", title_fontsize=7)
    fig.suptitle("self-vector mirror, by true speaker   " + title, fontsize=9)
    _save(fig, out_dir / "mirror_by_label.png")

    # 3. validity: is the model still answering, how hard was it pushed, and did
    # the self vector classify the tokens it reflected?
    fig, axes = plt.subplots(3, len(turns), figsize=(3.2 * len(turns), 7.2),
                             sharex=True, squeeze=False)
    for a, n in enumerate(turns):
        s = summary[int(n)]
        for c, mode in enumerate(modes):
            for kind, ls in (("self", "-"), ("random", "--")):
                if kind in kinds:
                    axes[0, a].plot(layers, s["cells"][kind][mode]["answer_mass"], ls,
                                    marker="o", ms=3, color=f"C{c}", label=f"{mode} {kind}")
            axes[1, a].plot(layers, s["cells"]["self"][mode]["disp"], "o-", ms=3,
                            color=f"C{c}", label=mode)
        axes[0, a].axhline(s["baseline"]["answer_mass"], color="k", ls=":", lw=1)
        axes[0, a].axhline(min_mass, color="C3", lw=0.8)
        axes[0, a].set(title=f"n = {n}", ylim=(0, 1.03))
        axes[2, a].plot(layers, s["z_acc_user"], "o-", ms=3, color="C2", label="user tokens")
        axes[2, a].plot(layers, s["z_acc_assistant"], "o-", ms=3, color="C1",
                        label="assistant tokens")
        axes[2, a].set(ylim=(0, 1.03), xlabel="mirrored layer k")
    axes[0, 0].set_ylabel("median P(User)+P(Assistant)")
    axes[1, 0].set_ylabel("median |x'-x| / |x|")
    axes[2, 0].set_ylabel("on own side of\nv.x+b=0 (pre-mirror)")
    axes[0, 0].legend(fontsize=6)
    axes[1, 0].legend(fontsize=6)
    axes[2, 0].legend(fontsize=6)
    fig.suptitle("validity  " + title, fontsize=9)
    _save(fig, out_dir / "mirror_validity.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", default=str(config.CAPTURE_PATH))
    p.add_argument("--vectors", default=str(config.VECTOR_PATH))
    p.add_argument("--mirror", default=str(config.MIRROR_PATH))
    p.add_argument("--out-dir", default=str(config.PLOT_DIR))
    args = p.parse_args()
    out_dir = Path(args.out_dir)

    cap, cmeta = capture.load(args.capture)
    vec = fit(cap)
    capture.save(args.vectors, vec, {**cmeta, "test_size": config.TEST_SIZE}, compress=False)
    print(f"saved {args.vectors}")
    print(" k  self  no-b  probe  cos(v,w)  |v|/|x|")
    for k in range(len(vec["acc"])):
        print(f"{k:2d}  {vec['acc'][k]:.3f} {vec['acc_nobias'][k]:.3f}  {vec['acc_lr'][k]:.3f}"
              f"  {vec['cos'][k]:+.3f}    {vec['v_norm'][k] / vec['x_norm'][k]:.3f}")
    plot_self_vector(vec, cmeta["model"], out_dir)

    results = {
        "model": cmeta["model"],
        "self_vector": {key: vec[key].tolist() for key in
                        ("acc", "acc_nobias", "acc_lr", "cos", "v_norm", "x_norm")},
    }
    if Path(args.mirror).exists():
        mir, mmeta = capture.load(args.mirror)
        if mmeta["model"] != cmeta["model"]:
            raise SystemExit(f"{args.mirror} is {mmeta['model']}, capture is {cmeta['model']}")
        summary = summarise(mir, mmeta)
        results["layers"] = mir["layers"].tolist()
        results["mirror"] = summary
        for n, s in summary.items():
            print(f"n={n}: unmirrored acc {s['baseline']['acc']:.3f}")
            for kind, by_mode in s["cells"].items():
                for mode, cell in by_mode.items():
                    print(f"  {kind:6s} {mode:9s} acc  "
                          + " ".join(f"{x:.2f}" for x in cell["acc"])
                          + "   flip " + " ".join(f"{x:.2f}" for x in cell["flip"]))
        plot_mirror(summary, mir, mmeta, out_dir)
    else:
        print(f"no {args.mirror} yet: run run_mirror, then analyze again for the mirror plots")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
