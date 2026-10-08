import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# built by claude

# outcome -> color; failed is neutral gray since the run produced nothing
OUTCOMES = [
    ("solved", "#2a78d6"),
    ("incorrect", "#eb6834"),
    ("failed", "#b4b4b4"),
]


def load(path):
    df = pd.read_csv(path)
    # rows with no n_tokens are runs that errored out: every endgame in them counts as failed
    df["failed_run"] = df["n_tokens"].isna()
    df["solved"] = df["n_solved"].where(~df["failed_run"], 0)
    df["failed"] = df["n_games"].where(df["failed_run"], 0)
    df["incorrect"] = df["n_games"] - df["solved"] - df["failed"]
    g = df.groupby(["model", "effort", "n_games"]).agg(
        runs=("n_solved", "size"), solved=("solved", "sum"), incorrect=("incorrect", "sum"), failed=("failed", "sum")
    ).reset_index()
    g["attempts"] = g["runs"] * g["n_games"]
    return df, g.sort_values(["model", "effort", "n_games"]).reset_index(drop=True)


def plot(g, out):
    groups = list(g.groupby(["model", "effort"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 0.45 * g["n_games"].nunique() + 1.8), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, ((model, effort), d) in zip(axes, groups):
        y = np.arange(len(d))
        left = np.zeros(len(d))
        for name, color in OUTCOMES:
            pct = (d[name] / d["attempts"] * 100).to_numpy()
            ax.barh(y, pct, left=left, height=0.75, color=color, edgecolor="white", linewidth=1.5, label=name)
            for yi, l, p in zip(y, left, pct):
                if p >= 8:
                    ax.text(l + p / 2, yi, f"{p:.0f}%", va="center", ha="center", fontsize=7, color="white")
            left += pct

        ax.set_yticks(y, [f"n_games={n}" for n in d["n_games"]], fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_title(f"{model} / {effort}  ({d['runs'].iloc[0]} runs per group)", fontsize=10)
        ax.set_xlabel("% of endgame attempts")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].invert_yaxis()  # y is shared, so invert once
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=len(OUTCOMES), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_tokens(df, out):
    groups = list(df.groupby(["model", "effort"]))
    levels = sorted(df["n_games"].unique())
    row = {n: i for i, n in enumerate(levels)}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 0.45 * len(levels) + 1.8), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)

    for ax, ((model, effort), d) in zip(axes, groups):
        ok = d[~d["failed_run"]]
        y = ok["n_games"].map(row) + rng.uniform(-0.2, 0.2, len(ok))
        ax.scatter(ok["n_tokens"] / 1000, y, s=18, color=OUTCOMES[0][1], alpha=0.7, linewidths=0)
        for n, t in ok.groupby("n_games")["n_tokens"]:
            ax.plot([t.median() / 1000] * 2, [row[n] - 0.32, row[n] + 0.32], color="#333", linewidth=1.5)
        for n, k in d.groupby("n_games")["failed_run"].sum().items():
            if k:
                ax.text(1.01, row[n], f"{k} failed", transform=ax.get_yaxis_transform(), va="center", fontsize=7, color="#777")

        ax.set_yticks(range(len(levels)), [f"n_games={n}" for n in levels], fontsize=8)
        ax.set_xlim(left=0)
        ax.set_title(f"{model} / {effort}", fontsize=10)
        ax.set_xlabel("n_tokens per run (thousands); bar = median, failed runs excluded")
        ax.grid(axis="x", color="#ddd", linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].invert_yaxis()  # y is shared, so invert once
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="out/endgames.csv")
    p.add_argument("--out", default="out/endgames.png")
    p.add_argument("--tokens-out", default="out/endgames_tokens.png")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    df, g = load(args.csv)
    print(g.to_string(index=False))
    plot(g, args.out)
    plot_tokens(df, args.tokens_out)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
