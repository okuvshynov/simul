import argparse
import glob
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Horizontal boxplots of turns and output tokens per variant
# (model-reasoning_effort[-note]),
# solved games only. Solved rate is shown in each box label.

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

def variant(r):
    parts = [r["model"], r["reasoning_effort"]]
    if r.get("note"):
        parts.append(r["note"])
    return "-".join(parts)

def boxplot(variants, values, labels, xlabel, title, path):
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(variants) + 1.5), dpi=150)
    ax.boxplot(
        values,
        vert=False,
        widths=0.5,
        showfliers=True,
        medianprops=dict(color="#1f2328", lw=2),
        boxprops=dict(color="#8b949e"),
        whiskerprops=dict(color="#8b949e"),
        capprops=dict(color="#8b949e"),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="#2a78d6",
                        markeredgecolor="none", alpha=0.7),
    )
    ax.set_yticks(range(1, len(variants) + 1), labels)
    # first variant on top
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)
    print(f"I: saved {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs/*.json")
    parser.add_argument("--out-prefix", default="boxplot")
    args = parser.parse_args()

    runs = [json.load(open(p)) for p in glob.glob(args.logs)]
    for r in runs:
        r["variant"] = variant(r)
    variants = sorted({r["variant"] for r in runs}, key=natural_key)

    solved = {v: [r for r in runs if r["variant"] == v and r["status"] == "solved"] for v in variants}
    total  = {v: sum(r["variant"] == v for r in runs) for v in variants}
    labels = [f"{v}\n{len(solved[v])}/{total[v]} solved ({100 * len(solved[v]) / total[v]:.0f}%)"
              for v in variants]

    boxplot(variants,
            [[r["n_turns"] for r in solved[v]] for v in variants],
            labels,
            "turns to solve",
            "Turns per solved game",
            f"{args.out_prefix}_turns.png")
    boxplot(variants,
            [[r["n_tokens_out_total"] / 1000 for r in solved[v]] for v in variants],
            labels,
            "total output tokens per game (thousands)",
            "Output tokens per solved game",
            f"{args.out_prefix}_tokens.png")

if __name__ == "__main__":
    main()
