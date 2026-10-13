import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# built by claude

DOT = "#2a78d6"
BAND = "#f2f2f2"


def load(path):
    df = pd.read_csv(path).dropna(subset=["n_tokens"])
    df["sample"] = df["path"].map(lambda p: Path(p).stem)
    # order samples by overall median so the between-sample spread reads top to bottom
    order = df.groupby("sample")["n_tokens"].median().sort_values().index.tolist()
    return df, order


def eta2(df, keys):
    """Share of total n_tokens variance explained by group means over `keys`."""
    t = df["n_tokens"]
    means = df.groupby(keys)["n_tokens"].transform("mean")
    return ((means - t.mean()) ** 2).sum() / ((t - t.mean()) ** 2).sum()


def plot(df, order, out):
    models = sorted(df["model"].unique())
    rows = [(s, m) for s in order for m in models if ((df["sample"] == s) & (df["model"] == m)).any()]
    ypos = {r: i for i, r in enumerate(rows)}
    rng = np.random.default_rng(0)
    pad = 5.2 * max(len(m) for m in models) + 14  # rough width of the model tick labels at fontsize 8

    fig, ax = plt.subplots(figsize=(10, 0.32 * len(rows) + 1.6))

    # alternating background band per sample
    for k, s in enumerate(order):
        ys = [ypos[r] for r in rows if r[0] == s]
        if k % 2 == 0:
            ax.axhspan(min(ys) - 0.5, max(ys) + 0.5, color=BAND, zorder=0, linewidth=0)
        # sample name sits left of the model tick labels
        ax.annotate(s, (0, (min(ys) + max(ys)) / 2), xycoords=ax.get_yaxis_transform(), xytext=(-pad, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=9, fontweight="bold")

    data = [df.loc[(df["sample"] == s) & (df["model"] == m), "n_tokens"].to_numpy() / 1000 for s, m in rows]
    ax.boxplot(data, positions=range(len(rows)), vert=False, widths=0.6, showfliers=False, zorder=2,
               medianprops=dict(color="#333", linewidth=1.5), boxprops=dict(color="#888"),
               whiskerprops=dict(color="#888"), capprops=dict(color="#888"))
    for i, d in enumerate(data):
        ax.scatter(d, i + rng.uniform(-0.22, 0.22, len(d)), s=14, color=DOT, alpha=0.6, linewidths=0, zorder=3)
        ax.text(1.005, i, f"n={len(d)}", transform=ax.get_yaxis_transform(), va="center", fontsize=7, color="#777")

    ax.set_yticks(range(len(rows)), [m for _, m in rows], fontsize=8)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(left=0)
    ax.set_xlabel("n_tokens per run (thousands); box = IQR, bar = median, whiskers = 1.5·IQR")
    ax.grid(axis="x", color="#ddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    e_s, e_sm = eta2(df, ["sample"]), eta2(df, ["sample", "model"])
    ax.set_title(f"n_tokens per run, by sample and model\nshare of variance explained: sample {e_s:.0%}, "
                 f"sample×model {e_sm:.0%}, within-group residual {1 - e_sm:.0%}", fontsize=10, loc="left")

    fig.tight_layout()
    fig.subplots_adjust(left=0.3)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="out/per_sample.csv")
    p.add_argument("--out", default="out/variance.png")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    df, order = load(args.csv)
    g = df.groupby(["sample", "model"])["n_tokens"].agg(["size", "mean", "median", "std"])
    g["cv"] = g["std"] / g["mean"]
    print(g.round(2).to_string())
    plot(df, order, args.out)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
