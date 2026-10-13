import argparse
import json
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator
from scipy import stats

# built by claude

# posterior lives in the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import noisy_mm  # noqa: E402
from greedy_player import posterior  # noqa: E402

# fixed categorical order; a model keeps its color whatever --models selects
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BAND = "#f2f2f2"


def valid_set_size(path):
    # valid_set_size is recorded before each guess, so the last turn's value is the candidate set
    # left after turns[:-1] -- exactly the history the model sees when making its final guess
    with open(path) as f:
        return json.load(f)["turns"][-1]["valid_set_size"]


def top_choice(path):
    """p of the most likely candidate after turns[:-1], and how many times likelier it is than the runner-up."""
    with open(path) as f:
        sample = json.load(f)
    history = [(t["guess"], *map(int, t["reply"].split())) for t in sample["turns"][:-1]]
    p = np.sort(np.array(list(posterior(noisy_mm.DATASET, history, sample["p_corr"]).values())))[::-1]
    p = p / p.sum()
    return p[0], (p[0] / p[1] if len(p) > 1 else np.inf)


def load(path, models=None):
    df = pd.read_csv(path).dropna(subset=["n_tokens"])
    df["sample"] = df["path"].map(lambda p: Path(p).stem)
    all_models = sorted(df["model"].unique())
    color = {m: COLORS[i % len(COLORS)] for i, m in enumerate(all_models)}
    if models:
        df = df[df["model"].isin(models)]
    df["log_t"] = np.log2(df["n_tokens"])
    # remove each model's overall scale: per-sample median in log space, averaged over samples,
    # so a model with fewer runs per sample still weighs every sample equally
    med = df.groupby(["model", "sample"])["log_t"].median()
    df["rel"] = df["log_t"] - df["model"].map(med.groupby("model").mean())
    order = df.groupby(["sample", "model"])["rel"].median().groupby("sample").mean().sort_values().index.tolist()
    vss = {Path(p).stem: valid_set_size(p) for p in df["path"].unique()}
    top = {Path(p).stem: top_choice(p) for p in df["path"].unique()}
    return df, order, color, vss, top


def decompose(df):
    """Sequential sums of squares for log2 n_tokens: model, then sample, then sample×model, residual."""
    y = df["log_t"].to_numpy()
    terms = [("model", ["model"]), ("sample", ["model", "sample"]), ("sample×model", ["model", "sample", "inter"])]
    d = df.assign(inter=df["sample"] + "|" + df["model"])
    total = ((y - y.mean()) ** 2).sum()
    prev, out = total, {}
    for name, cols in terms:
        X = pd.get_dummies(d[cols], drop_first=True, dtype=float).to_numpy()
        X = np.column_stack([np.ones(len(y)), X])
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        ss = (resid ** 2).sum()
        out[name] = (prev - ss) / total
        prev = ss
    out["within"] = prev / total
    return out


def concordance(df):
    """Per-sample median rel per model; Spearman between model pairs and Kendall's W across models."""
    m = df.groupby(["sample", "model"])["rel"].median().unstack()
    m = m.dropna()  # only samples every model ran
    rho = m.corr(method="spearman")
    k, n = m.shape[1], m.shape[0]
    R = m.rank().sum(axis=1)
    W = 12 * ((R - R.mean()) ** 2).sum() / (k ** 2 * (n ** 3 - n))
    return m, rho, W


def metric_corr(df, metric):
    """Spearman between a per-sample metric and per-sample median rel, per model and for the model average."""
    m = df.groupby(["sample", "model"])["rel"].median().unstack()
    m["mean"] = m.mean(axis=1)
    v = m.index.map(metric)
    return {c: stats.spearmanr(v, m[c], nan_policy="omit").statistic for c in m.columns}


def set_xaxis(ax, x, raw, note):
    """Ticks for log2 values: token counts when raw, multipliers of the model's typical run otherwise."""
    lo, hi = x.min(), x.max()
    if raw:
        nice = np.sort((np.array([1, 2, 5]) * 10.0 ** np.arange(2, 7)[:, None]).ravel())
        nice = nice[(np.log2(nice) >= lo - 0.5) & (np.log2(nice) <= hi + 0.5)]
        ticks, labels = np.log2(nice), [f"{n / 1000:g}k" for n in nice]
        xlabel = f"n_tokens per run (log scale); {note}"
    else:
        ax.axvline(0, color="#555", linewidth=1, zorder=1)
        ticks = np.arange(np.floor(lo), np.ceil(hi) + 1)
        labels = [f"×{2 ** t:g}" if t >= 0 else f"×1/{2 ** -t:g}" for t in ticks]
        xlabel = f"n_tokens relative to the model's typical run (log scale); {note}"
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(labels)
    ax.set_xlim(lo - 0.15, hi + 0.15)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#ddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def plot_scatter(df, order, color, vss, out, raw=False):
    """One row per sample, all models' runs in the same row; easiest sample at the bottom."""
    models = sorted(df["model"].unique())
    col = "log_t" if raw else "rel"
    row = {s: i for i, s in enumerate(order)}
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(order) + 1.6))
    for i in range(0, len(order), 2):
        ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0, linewidth=0)
    for m in models:
        d = df[df["model"] == m]
        ax.scatter(d[col], d["sample"].map(row) + rng.uniform(-0.25, 0.25, len(d)), s=16, color=color[m],
                   alpha=0.7, linewidths=0, zorder=3, label=m)
        for s, v in d.groupby("sample")[col].median().items():
            ax.plot([v, v], [row[s] - 0.35, row[s] + 0.35], color=color[m], linewidth=2.5, zorder=4,
                    path_effects=[pe.Stroke(linewidth=4.5, foreground="white"), pe.Normal()])

    set_xaxis(ax, df[col], raw, "bar = median")
    ax.set_yticks(range(len(order)), [f"{s}\nvalid set size: {vss[s]}" for s in order], fontsize=8)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(-0.5, len(order) - 0.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(models), frameon=False, fontsize=8,
              handletextpad=0.2, markerscale=1.5)
    ax.set_title("n_tokens per sample" if raw else "normalized n_tokens per sample", fontsize=10, loc="left", pad=22)

    fig.subplots_adjust(left=0.17, right=0.97, top=1 - 0.6 / fig.get_figheight(), bottom=0.7 / fig.get_figheight())
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot(df, order, color, vss, top, out, raw=False, by_model=False):
    models = sorted(df["model"].unique())
    has = set(zip(df["sample"], df["model"]))
    # rows are (sample, model); the outer grouping is by sample, or by model with --by-model
    if by_model:
        groups = models
        # easiest sample at the bottom of each block, so medians step up and to the right
        rows = [(s, m) for m in models for s in reversed(order) if (s, m) in has]
        group_of = lambda r: r[1]
        tick = lambda r: (f"{r[0]}\nvalid set {vss[r[0]]} · p_top {top[r[0]][0]:.2f}"
                          f" · top/2nd ×{top[r[0]][1]:.1f}")
    else:
        groups = order
        rows = [(s, m) for s in order for m in models if (s, m) in has]
        group_of, tick = (lambda r: r[0]), (lambda r: r[1])
    rng = np.random.default_rng(0)
    # rough width of the widest tick label line at fontsize 8
    pad = 5.2 * max(len(line) for r in rows for line in tick(r).split("\n")) + 14

    fig, ax = plt.subplots(figsize=(10, (0.42 if by_model else 0.3) * len(rows) + 2.2))

    for k, g in enumerate(groups):
        ys = [i for i, r in enumerate(rows) if group_of(r) == g]
        if k % 2 == 0:
            ax.axhspan(min(ys) - 0.5, max(ys) + 0.5, color=BAND, zorder=0, linewidth=0)
        mid = (min(ys) + max(ys)) / 2
        if by_model:
            # vertical, since the sample tick labels already take the horizontal room
            ax.annotate(g, (0, mid), xycoords=ax.get_yaxis_transform(), xytext=(-pad, 0), rotation=90,
                        textcoords="offset points", ha="right", va="center", fontsize=9, fontweight="bold")
            continue
        ax.annotate(g, (0, mid), xycoords=ax.get_yaxis_transform(), xytext=(-pad, 5),
                    textcoords="offset points", ha="right", va="bottom", fontsize=9, fontweight="bold")
        ax.annotate(f"valid set size: {vss[g]}", (0, mid), xycoords=ax.get_yaxis_transform(), xytext=(-pad, 3),
                    textcoords="offset points", ha="right", va="top", fontsize=8, color="#555")

    # both are log2 values, so boxes and dots are drawn the same way; only the ticks differ
    col = "log_t" if raw else "rel"
    data = [df.loc[(df["sample"] == s) & (df["model"] == m), col].to_numpy() for s, m in rows]
    ax.boxplot(data, positions=range(len(rows)), vert=False, widths=0.6, showfliers=False, zorder=2,
               medianprops=dict(color="#333", linewidth=1.5), boxprops=dict(color="#999"),
               whiskerprops=dict(color="#999"), capprops=dict(color="#999"))
    for i, ((s, m), d) in enumerate(zip(rows, data)):
        ax.scatter(d, i + rng.uniform(-0.22, 0.22, len(d)), s=14, color=color[m], alpha=0.7, linewidths=0, zorder=3)
        ax.text(1.005, i, f"n={len(d)}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7, color="#777")

    set_xaxis(ax, df[col], raw, "box = IQR, bar = median")
    ax.set_yticks(range(len(rows)), [tick(r) for r in rows], fontsize=8)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(len(rows) - 0.5, -0.5)

    ax.set_title("n_tokens per sample" if raw else "normalized n_tokens per sample", fontsize=10, loc="left")

    # leave room for the tick labels plus the group label to their left (pad is in points, 72 per inch)
    left = max(0.27, (pad + 28) / (72 * fig.get_figwidth()))
    fig.subplots_adjust(left=left, right=0.95, top=1 - 0.45 / fig.get_figheight(), bottom=0.7 / fig.get_figheight())
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="out/per_sample.csv")
    p.add_argument("--out", default="out/variance.png")
    p.add_argument("--models", nargs="+", help="only these models (default: all)")
    p.add_argument("--raw", action="store_true", help="plot raw n_tokens on a log scale instead of normalized per model")
    p.add_argument("--by-model", action="store_true", help="group rows by model instead of by sample")
    p.add_argument("--scatter", action="store_true", help="one row per sample with all models' runs as dots, no boxes")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    df, order, color, vss, top = load(args.csv, args.models)
    g = df.groupby(["sample", "model"])["n_tokens"].agg(["size", "median"]).unstack()
    print(g.to_string(), "\n")
    shares = decompose(df)
    print("share of log2 n_tokens variance:", {k: round(v, 3) for k, v in shares.items()})
    m, rho, W = concordance(df)
    print("\nper-sample median, relative to model (log2):\n", m.round(2).to_string())
    print("\nspearman between models (sample medians):\n", rho.round(2).to_string())
    print(f"\nKendall's W = {W:.3f} over {m.shape[0]} samples, {m.shape[1]} models")
    print("\nper sample: valid set size, p_top, top/2nd")
    for s in order:
        print(f"  {s}  {vss[s]:4d}  {top[s][0]:.3f}  {top[s][1]:.2f}")
    # higher p_top / ratio means an easier choice, so flip the sign to compare with difficulty
    for name, metric in [("valid set size", vss), ("-p_top", {s: -v[0] for s, v in top.items()}),
                         ("-top/2nd", {s: -v[1] for s, v in top.items()})]:
        rho = metric_corr(df, metric)
        print(f"spearman({name}, sample median rel):", {k: round(v, 2) for k, v in rho.items()})
    if args.scatter:
        plot_scatter(df, order, color, vss, args.out, args.raw)
    else:
        plot(df, order, color, vss, top, args.out, args.raw, args.by_model)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
