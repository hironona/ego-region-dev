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


def steering_vectors(contrast_X, contrast_y, contrast_set, n_sets):
    """(S, L+1, D): per donor persona, per layer, mean(trait) - mean(no trait).

    Unnormalised, so a coefficient of 1.0 means "one full persona shift" — the
    usual CAA convention, and what makes alpha* comparable to the sweep grid.
    """
    X = contrast_X.astype(np.float64)
    out = np.zeros((n_sets, X.shape[1], X.shape[2]))
    for si in range(n_sets):
        sel = contrast_set == si
        assert sel.any(), f"no rows for vector set {si}"
        out[si] = X[sel & (contrast_y == 1)].mean(0) - X[sel & (contrast_y == 0)].mean(0)
    return out


def target_direction(Vs, names, eval_set):
    """(L+1, D) the direction that moves the model *toward* the scored answer.

    Mind the sign. A persona vector points toward that persona's own
    answer_matching_behavior, but the eval scores answer_NOT_matching_behavior —
    the disagreeable choice on an agreeableness eval. So the direction we want
    donors to align with is the eval persona's vector NEGATED. Skipping this
    negation flips every relevance score and turns "most relevant donor" into
    "least relevant", which is the kind of error that survives a plot.
    """
    assert eval_set in names, (
        f"{eval_set!r} not among captured vector sets {names}; re-run run_capture "
        "with it included, or relevance has nothing to measure against."
    )
    return -Vs[names.index(eval_set)]


def relevance(Vs, names, eval_set):
    """(S, L+1) cosine of each donor vector against the eval's target direction.

    This is the measurement the hand-picked "anger"/"refusal" traits were missing.
    A donor steers this eval only to the extent that its trait direction overlaps
    the axis the eval's own items vary along; anger has no such overlap with
    agreeableness statements, which is why that sweep was flat.

    Positive means a positive steering coefficient pushes toward the scored
    answer. The eval persona itself scores exactly -1.0 (it is the negated
    reference) — that entry is the scale check, not a candidate.
    """
    ref = target_direction(Vs, names, eval_set)
    num = np.einsum("skd,kd->sk", Vs, ref)
    den = np.linalg.norm(Vs, axis=2) * np.linalg.norm(ref, axis=1)[None, :]
    return num / (den + 1e-12)


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


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_steering(steer, alpha_star, out_dir):
    """The requested figure: steering effect vs coefficient, one panel per steered
    layer, with the speaker boundary crossing marked.

    The dashed vertical line is the median alpha* over eval prompts (the band is
    the interquartile range); the star sits on the curve at that coefficient, so
    the vertical-axis reading is "steering effect at the moment the prompt crosses
    the learned user/assistant boundary".
    """
    layers, coeffs, acc = steer["layers"], steer["coeffs"], steer["acc"]
    ncols = 3
    nrows = int(np.ceil(len(layers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)

    for i, k in enumerate(layers):
        ax = axes.flat[i]
        ax.plot(coeffs, acc[i], "o-", color="tab:purple", lw=1.6, ms=4, label="target choice")
        ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")

        a = alpha_star[k]
        lo, med, hi = np.nanpercentile(a, [25, 50, 75])
        ax.axvspan(lo, hi, color="tab:green", alpha=0.13)
        ax.axvline(med, color="tab:green", ls="--", lw=1.4, label="speaker boundary (z=0)")
        y_at = np.interp(med, coeffs, acc[i], left=np.nan, right=np.nan)
        if np.isfinite(y_at):
            ax.plot([med], [y_at], "*", color="tab:green", ms=16, mec="k", mew=0.5, zorder=5)
            ax.annotate(
                f"{y_at:.2f} @ a*={med:.1f}",
                (med, y_at), textcoords="offset points", xytext=(8, 8), fontsize=8,
            )
        else:
            ax.annotate(
                f"a*={med:.1f} (off grid)", (0.02, 0.92), xycoords="axes fraction", fontsize=8
            )

        ax.set_title(f"steer at layer {k}", fontsize=11)
        ax.set(xlabel="steering coefficient", ylabel="target-choice accuracy", ylim=(-0.02, 1.02))
        ax.grid(alpha=0.25)

    for ax in axes.flat[len(layers):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"{steer_meta_title(steer)} — steering effect vs the speaker boundary", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out_dir / "steering_effect_vs_boundary.png")


def steer_meta_title(steer):
    return f"vector={steer['vector_set']}  eval={steer['eval_set']}"


def plot_relevance(rel, names, eval_set, layers, out_dir):
    """Per-layer cosine of every donor vector against the eval's own vector.

    Read this before believing any steering curve: a donor flat against the
    reference has no axis in the eval to move along, and a null result from it
    says nothing about steering — only that the trait was irrelevant.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ks = np.arange(rel.shape[1])
    for si, name in enumerate(names):
        is_ref = name == eval_set  # the negated reference: pinned at -1 by construction
        ax.plot(
            ks, rel[si], "o-", ms=3, lw=2.0 if is_ref else 1.4,
            color="k" if is_ref else None,
            ls="--" if is_ref else "-",
            label=f"{name} (negated reference)" if is_ref else name,
        )
    ax.axhline(0, color="gray", ls=":", lw=1)
    for k in layers:
        ax.axvline(k, color="k", lw=0.5, alpha=0.25)
    ax.set(
        title=f"donor-trait relevance to {eval_set}",
        xlabel="layer", ylabel=f"cosine with the {eval_set} vector",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / "vector_relevance.png")


def plot_geometry(acc, cos, alpha_star, layers, out_dir):
    """Why the crossing sits where it does: probe quality, the angle between the
    probe normal and the steering vector, and alpha* itself, all per layer."""
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
    axes[2].set(title="boundary crossing coefficient a*", xlabel="layer", ylabel="a*")

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, out_dir / "boundary_geometry.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", default=str(config.CAPTURE_PATH))
    p.add_argument("--vectors", default=str(config.VECTOR_PATH))
    p.add_argument("--steer", default=str(config.STEER_PATH))
    p.add_argument("--plots", default=str(config.PLOT_DIR))
    p.add_argument("--vector-set", default=config.VECTOR_SET,
                   help="which captured donor persona to analyse and plot")
    args = p.parse_args()

    arrays, meta = capture.load(args.capture)
    W, B, acc = fit_speaker_boundary(
        arrays["speaker_X"], arrays["speaker_y"], arrays["speaker_conv"], seed=meta["seed"]
    )
    names = list(meta["vector_sets"])
    Vs = steering_vectors(
        arrays["contrast_X"], arrays["contrast_y"], arrays["contrast_set"], len(names)
    )
    rel = (
        relevance(Vs, names, meta["eval_set"])
        if meta["vector_source"] == "persona" and meta["eval_set"] in names
        else np.full((len(names), Vs.shape[1]), np.nan)
    )

    chosen = args.vector_set if args.vector_set in names else names[0]
    if chosen != args.vector_set:
        print(f"--vector-set {args.vector_set!r} not captured; using {chosen!r} of {names}")
    V = Vs[names.index(chosen)]
    alpha_star, z0, denom = crossings(arrays["eval_X"], W, B, V)
    cos = cosines(W, V)

    capture.save(
        args.vectors,
        {
            "W": W, "B": B, "probe_acc": acc, "V": V, "Vs": Vs, "relevance": rel,
            "alpha_star": alpha_star, "z0": z0, "cos": cos,
        },
        {**meta, "derived_from": str(args.capture), "vector_set": chosen},
        compress=False,
    )
    print(f"saved {args.vectors}")
    print(f"probe acc: min {acc.min():.3f} mean {acc.mean():.3f} max {acc.max():.3f}")

    if np.isfinite(rel).any():
        mid = slice(Vs.shape[1] // 3, 2 * Vs.shape[1] // 3)  # middle third of layers
        print(f"\nrelevance to the {meta['eval_set']} target answer, mean over")
        print("middle-third layers. Positive => a positive coefficient steers toward")
        print("the scored choice; magnitude => how much axis the donor shares at all:")
        for si, name in sorted(
            enumerate(names), key=lambda t: -abs(np.nanmean(rel[t[0], mid]))
        ):
            note = " (negated reference, not a candidate)" if name == meta["eval_set"] else ""
            note += " <- steering" if name == chosen else ""
            print(f"  {np.nanmean(rel[si, mid]):+.3f}  {name}{note}")

    print(f"\nsteering with {chosen!r}")
    print("layer  probe_acc  cos(w,v)  median a*  |v|")
    for k in config.STEER_LAYERS:
        print(
            f"{k:5d}  {acc[k]:9.3f}  {cos[k]:8.3f}  {np.nanmedian(alpha_star[k]):9.2f}"
            f"  {np.linalg.norm(V[k]):6.2f}"
        )

    out_dir = Path(args.plots)
    steer_path = Path(args.steer)
    if not steer_path.exists():
        print(f"\n{steer_path} not found -- run run_steer.py, then re-run this to plot.")
        return

    steer_arrays, steer_meta = capture.load(steer_path)
    steer = {
        **steer_arrays,
        "vector_set": steer_meta.get("vector_set", chosen),
        "eval_set": steer_meta["eval_set"],
    }
    assert steer["vector_set"] == chosen, (
        f"steer.npz was swept with vector {steer['vector_set']!r} but this run "
        f"analysed {chosen!r}; pass --vector-set {steer['vector_set']} or re-sweep."
    )
    plot_steering(steer, alpha_star, out_dir)
    plot_geometry(acc, cos, alpha_star, steer["layers"], out_dir)
    if np.isfinite(rel).any():
        plot_relevance(rel, names, meta["eval_set"], steer["layers"], out_dir)

    summary = {
        "vector_set": chosen,
        "vector_sets": names,
        "relevance": rel.tolist(),
        "probe_acc": acc.tolist(),
        "cos_w_v": cos.tolist(),
        "median_alpha_star": np.nanmedian(alpha_star, axis=1).tolist(),
        "steer_layers": steer["layers"].tolist(),
        "coeffs": steer["coeffs"].tolist(),
        "target_choice_acc": steer["acc"].tolist(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
