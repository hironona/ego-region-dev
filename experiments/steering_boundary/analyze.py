"""capture.npz -> speaker boundary + steering vectors (vectors.npz), then
steer.npz -> plots. Never loads the model.

Two stages because run_steer.py sits between them:

    uv run python -m experiments.steering_boundary.analyze   # writes vectors.npz
    uv run python -m experiments.steering_boundary.run_steer # writes steer.npz
    uv run python -m experiments.steering_boundary.analyze   # writes plots

The second call re-derives vectors.npz identically (it is cheap and deterministic)
and additionally plots, so there is no stage flag to get wrong.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from core import capture  # noqa: E402

from . import config  # noqa: E402


def fit_speaker_boundary(X, y, conv, seed=0, test_size=0.25):
    """Per layer: logistic speaker probe, returned as a raw-space hyperplane.

    sklearn fits on standardised features; undoing the scaler here gives (w, b)
    such that z = X @ w + b is the same decision value in the units the residual
    stream (and therefore the steering vector) actually lives in. Without that
    undo, alpha* would be computed in one space and applied in another.

    Split is by conversation so the held-out accuracy is not leaked across turns.
    """
    n_layers, d = X.shape[1], X.shape[2]
    idx = np.arange(len(conv))
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(
            idx, groups=conv
        )
    )
    W = np.zeros((n_layers, d))
    B = np.zeros(n_layers)
    acc = np.zeros(n_layers)
    for k in range(n_layers):
        Xk = X[:, k, :].astype(np.float64)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(Xk[train_idx], y[train_idx])
        scaler, lr = clf[0], clf[-1]
        W[k] = lr.coef_[0] / scaler.scale_
        B[k] = lr.intercept_[0] - float((lr.coef_[0] * scaler.mean_ / scaler.scale_).sum())
        preds = clf.predict(Xk[test_idx])
        yt = y[test_idx]
        acc[k] = np.mean([(preds[yt == c] == c).mean() for c in (0, 1)])
    return W, B, acc


def steering_vectors(contrast_X, contrast_y):
    """(L+1, D) per layer: mean(label 1) - mean(label 0) of the contrast.

    Label 1 is the trait side: the trait answer token for "caa", or the trait
    system prompt for "system_prompt". The vector is kept unnormalised. The sweep
    sets its size through h_norm / |v| (see config.STEER_SCALES), so the length
    of v only matters for converting alpha* onto that grid.
    """
    X = contrast_X.astype(np.float64)
    return X[contrast_y == 1].mean(0) - X[contrast_y == 0].mean(0)


def crossings(eval_X, W, B, V):
    """(L+1, N) coefficient at which each eval prompt crosses z = 0.

    Steering adds alpha*v to hidden_states[k], so z(alpha) = z0 + alpha*(w.v)
    exactly, and alpha* = -z0 / (w.v). NaN where w and v are near-orthogonal,
    i.e. where steering does not move the speaker boundary at all.
    """
    X = eval_X.astype(np.float64)
    z0 = np.einsum("nkd,kd->kn", X, W) + B[:, None]
    denom = np.einsum("kd,kd->k", W, V)
    scale = np.linalg.norm(W, axis=1) * np.linalg.norm(V, axis=1)
    denom = np.where(np.abs(denom) > 1e-9 * np.maximum(scale, 1e-12), denom, np.nan)
    return -z0 / denom[:, None], z0, denom


def cosines(W, V):
    return np.einsum("kd,kd->k", W, V) / (
        np.linalg.norm(W, axis=1) * np.linalg.norm(V, axis=1) + 1e-12
    )


def read_position_norms(eval_X):
    """(L+1,) median |h_k| at the read position of the unsteered eval prompts.

    This is the unit of the sweep grid: scale r means |c * v_k| = r * h_norm[k].
    It is measured at the read position only. Other positions (the attention
    sink in particular) can be far larger, so r is "relative to the activation
    the answer is read from", not to every token.
    """
    return np.median(np.linalg.norm(eval_X.astype(np.float64), axis=-1), axis=0)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _broken_spans(ax, scales, broken):
    """Grey out the stretch of the x axis where the model stopped answering.

    Each broken grid point owns the interval halfway to its neighbours, so the
    bands tile the axis on a non-uniform grid.
    """
    mid = (scales[1:] + scales[:-1]) / 2
    edge = np.diff(scales[[0, 1]])[0] / 2 if len(scales) > 1 else 0.5
    lo = np.concatenate([[scales[0] - edge], mid])
    hi = np.concatenate([mid, [scales[-1] + edge]])
    for a, b in zip(lo[broken], hi[broken]):
        ax.axvspan(a, b, color="0.88", lw=0, zorder=0)


def plot_steering(steer, alpha_scale, out_dir):
    """Steering effect vs scale, one panel per steered layer, against the speaker
    boundary and a random direction of the same norm.

    The behavioural readout is split by answer key, because the unsplit accuracy
    cannot tell the two effects apart on a 50%-Yes key:
      trait effect  pushes the callous answer on *both* halves: acc|target=Yes
                    and acc|target=No rise together.
      answer bias   "say Yes" or "say No" to everything: one half goes to 1, the
                    other stays at 0. The mean is exactly 0.5, which is what the
                    earlier plots read as a steering effect.
    Solid lines are the trait vector and faint grey lines the random control.
    Grey bands mark cells where the model is not answering any more (median
    P(Yes)+P(No) < config.MIN_ANSWER_MASS). Accuracy there is an argmax between
    two tail tokens and is not a measurement.

    How the speaker boundary is drawn depends on what was steered:
    "last"  one activation per prompt is pushed, so each prompt has one exact
            crossing, alpha*, drawn as a dashed line at its median (in scale
            units) with the IQR shaded.
    "all"   every position is pushed and each crosses at its own alpha, so the
            boundary is drawn as the fraction of steered activations on the
            assistant side of z = 0 instead.
    """
    layers, scales = steer["layers"], steer["scales"]
    all_positions = steer["steer_positions"] == "all"
    kinds = list(steer["kinds"])
    ncols = 3
    nrows = int(np.ceil(len(layers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), squeeze=False)

    for i, k in enumerate(layers):
        ax = axes.flat[i]
        t = kinds.index("trait")
        broken = steer["answer_mass"][t, i] < steer["min_answer_mass"]
        _broken_spans(ax, scales, broken)
        ax.axhline(0.5, color="gray", ls=":", lw=1)

        if "random" in kinds:
            r = kinds.index("random")
            ax.plot(scales, steer["acc_tgt_yes"][r, i], "-", color="0.6", lw=1, alpha=0.8,
                    label="random: acc | target=Yes")
            ax.plot(scales, steer["acc_tgt_no"][r, i], "--", color="0.6", lw=1, alpha=0.8,
                    label="random: acc | target=No")
        ax.plot(scales, steer["acc_tgt_yes"][t, i], "o-", color="tab:purple", lw=1.6, ms=4,
                label="trait: acc | target=Yes")
        ax.plot(scales, steer["acc_tgt_no"][t, i], "s--", color="tab:purple", lw=1.6, ms=4,
                mfc="white", label="trait: acc | target=No")

        if all_positions:
            ax.plot(scales, steer["crossed"][t, i], "^-", color="tab:green", lw=1.2, ms=3.5,
                    label="fraction past z=0")
        else:
            a = alpha_scale[k]
            lo, med, hi = np.nanpercentile(a, [25, 50, 75])
            ax.axvspan(lo, hi, color="tab:green", alpha=0.13)
            ax.axvline(med, color="tab:green", ls="--", lw=1.4, label="speaker boundary (z=0)")
            if not scales.min() <= med <= scales.max():
                ax.annotate(f"a*={med:.2f} (off grid)", (0.02, 0.92),
                            xycoords="axes fraction", fontsize=8)

        ax.set_title(f"steer at layer {k}", fontsize=11)
        ax.set(xlabel="steering scale  |c v| / |h|", ylabel="fraction", ylim=(-0.02, 1.02))
        ax.grid(alpha=0.25)

    for ax in axes.flat[len(layers):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    fig.suptitle(
        f"{steer_meta_title(steer)} — steering effect vs the speaker boundary "
        f"(grey band: not answering)", fontsize=12
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    _save(fig, out_dir / "steering_effect_vs_boundary.png")


def plot_validity(steer, out_dir):
    """Is each cell still a Yes/No answer, and which way does it lean?

    Yes-rate at 0 or 1 means one constant answer. Answer mass near 0 means the
    readout is comparing two tail logits. Both effects give an accuracy of 0.5,
    so this is the plot to check before reading steering_effect_vs_boundary.png.
    """
    layers, scales, kinds = steer["layers"], steer["scales"], list(steer["kinds"])
    fig, axes = plt.subplots(2, len(layers), figsize=(4.2 * len(layers), 6.2), squeeze=False)
    style = {"trait": dict(color="tab:purple", lw=1.6), "random": dict(color="0.55", lw=1.2)}
    for i, k in enumerate(layers):
        for j, kind in enumerate(kinds):
            axes[0, i].plot(scales, steer["yes_rate"][j, i], "o-", ms=3.5, label=kind, **style[kind])
            axes[1, i].plot(scales, steer["answer_mass"][j, i], "o-", ms=3.5, label=kind,
                            **style[kind])
        axes[0, i].axhline(0.5, color="gray", ls=":", lw=1)
        axes[1, i].axhline(steer["min_answer_mass"], color="tab:red", ls=":", lw=1)
        axes[0, i].set(title=f"layer {k}: Yes-rate", ylim=(-0.02, 1.02))
        axes[1, i].set(title=f"layer {k}: median P(Yes)+P(No)", ylim=(-0.02, 1.02),
                       xlabel="steering scale  |c v| / |h|")
        for ax in axes[:, i]:
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{steer_meta_title(steer)} — is the model still answering?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_dir / "steering_validity.png")


def steer_meta_title(steer):
    return (f"vector={steer['vector_method']} ({steer['trait']})  eval={steer['eval_set']}  "
            f"{steer['steer_positions']} positions")


def plot_geometry(acc, cos, alpha_star, layers, out_dir):
    """Why the crossing sits where it does: probe quality, the angle between the
    probe normal and the steering vector, and alpha* itself, all per layer.
    `alpha_star` is expected in scale units (alpha_scale), the sweep's x axis."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    ks = np.arange(len(acc))

    axes[0].plot(ks, acc, "o-", ms=3, color="tab:blue")
    axes[0].axhline(0.5, color="gray", ls=":")
    axes[0].set(title="speaker probe accuracy", xlabel="layer", ylabel="balanced acc", ylim=(0.4, 1.02))

    axes[1].plot(ks, cos, "o-", ms=3, color="tab:orange")
    axes[1].axhline(0, color="gray", ls=":")
    axes[1].set(title="cos(probe normal, steering vector)", xlabel="layer", ylabel="cosine")

    med = np.nanmedian(alpha_star, axis=1)
    lo, hi = np.nanpercentile(alpha_star, 25, axis=1), np.nanpercentile(alpha_star, 75, axis=1)
    axes[2].fill_between(ks, lo, hi, alpha=0.2, color="tab:green")
    axes[2].plot(ks, med, "o-", ms=3, color="tab:green")
    axes[2].axhline(0, color="gray", ls=":")
    for k in layers:
        axes[2].axvline(k, color="k", lw=0.5, alpha=0.3)
    axes[2].set(title="boundary crossing a* (scale units)", xlabel="layer",
                ylabel="a* |v| / |h|")

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, out_dir / "boundary_geometry.png")


STEER_KEYS = ("scales", "coeffs", "acc", "acc_tgt_yes", "acc_tgt_no", "yes_rate",
              "answer_mass", "margin", "crossed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", default=str(config.CAPTURE_PATH))
    p.add_argument("--vectors", default=str(config.VECTOR_PATH))
    p.add_argument("--steer", default=str(config.STEER_PATH))
    p.add_argument("--plots", default=str(config.PLOT_DIR))
    args = p.parse_args()

    arrays, meta = capture.load(args.capture)
    # Captures from before VECTOR_METHOD existed are system-prompt captures.
    method = meta.get("vector_method", "system_prompt")
    if method != config.VECTOR_METHOD:
        raise SystemExit(
            f"{args.capture} holds a {method!r} contrast but config.VECTOR_METHOD is "
            f"{config.VECTOR_METHOD!r}. Re-run run_capture (FORCE_CAPTURE=1 in the "
            f"slurm job), or pass --vector-method {method} to run_capture next time "
            f"and set VECTOR_METHOD to match."
        )
    W, B, acc = fit_speaker_boundary(
        arrays["speaker_X"], arrays["speaker_y"], arrays["speaker_conv"], seed=meta["seed"]
    )
    V = steering_vectors(arrays["contrast_X"], arrays["contrast_y"])
    alpha_star, z0, denom = crossings(arrays["eval_X"], W, B, V)
    cos = cosines(W, V)
    h_norm = read_position_norms(arrays["eval_X"])
    v_norm = np.linalg.norm(V, axis=1)
    # alpha* on the sweep's axis: the scale r at which prompt n crosses z = 0.
    alpha_scale = alpha_star * (v_norm / h_norm)[:, None]

    capture.save(
        args.vectors,
        {
            "W": W, "B": B, "probe_acc": acc, "V": V, "h_norm": h_norm,
            "alpha_star": alpha_star, "alpha_scale": alpha_scale, "z0": z0, "cos": cos,
        },
        {**meta, "vector_method": method, "derived_from": str(args.capture)},
        compress=False,
    )
    print(f"saved {args.vectors}")
    print(f"probe acc: min {acc.min():.3f} mean {acc.mean():.3f} max {acc.max():.3f}")

    print(f"\nsteering vector={method} ({meta['trait']}) on eval={meta['eval_set']!r}")
    print("layer  probe_acc  cos(w,v)  median a*  |v|/|h|   |v|    |h|")
    for k in config.STEER_LAYERS:
        print(
            f"{k:5d}  {acc[k]:9.3f}  {cos[k]:8.3f}  {np.nanmedian(alpha_scale[k]):9.2f}"
            f"  {v_norm[k] / h_norm[k]:7.3f}  {v_norm[k]:6.2f}  {h_norm[k]:6.1f}"
        )
    print("(median a* is in scale units |a v|/|h|, the sweep's x axis)")

    out_dir = Path(args.plots)
    steer_path = Path(args.steer)
    if not steer_path.exists():
        print(f"\n{steer_path} not found -- run run_steer.py, then re-run this to plot.")
        return

    steer_arrays, steer_meta = capture.load(steer_path)
    missing = [key for key in STEER_KEYS if key not in steer_arrays]
    if missing:
        raise SystemExit(f"{steer_path} predates the split readout (missing {missing}); "
                         f"re-run run_steer.")
    if steer_meta.get("vector_method", "system_prompt") != method:
        raise SystemExit(f"{steer_path} steered a {steer_meta.get('vector_method')!r} "
                         f"vector, but {args.capture} is {method!r}; re-run run_steer.")
    steer = {
        **steer_arrays,
        "trait": steer_meta["trait"],
        "eval_set": steer_meta["eval_set"],
        "steer_positions": steer_meta["steer_positions"],
        "vector_method": method,
        "kinds": steer_meta["kinds"],
        "min_answer_mass": steer_meta["min_answer_mass"],
    }
    plot_steering(steer, alpha_scale, out_dir)
    plot_validity(steer, out_dir)
    plot_geometry(acc, cos, alpha_scale, steer["layers"], out_dir)

    summary = {
        "vector_method": method,
        "trait": meta["trait"],
        "steer_positions": steer["steer_positions"],
        "probe_acc": acc.tolist(),
        "cos_w_v": cos.tolist(),
        "v_over_h": (v_norm / h_norm).tolist(),
        "median_alpha_star_scale": np.nanmedian(alpha_scale, axis=1).tolist(),
        "steer_layers": steer["layers"].tolist(),
        "kinds": steer["kinds"],
        "min_answer_mass": steer["min_answer_mass"],
    }
    # Metric arrays are (kind, layer, scale). coeffs is (layer, scale): the raw
    # coefficient behind each scale, shared by both kinds since they have the same norm.
    summary.update({key: steer[key].tolist() for key in STEER_KEYS})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
