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

# categorical hues assigned to models in fixed (sorted) order; boxes get a light tint
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BOX_FILL_ALPHA = 0.2

def boxplot(variants, values, labels, colors, xlabel, title, path):
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(variants) + 1.5), dpi=150)
    bp = ax.boxplot(
        values,
        vert=False,
        widths=0.5,
        showfliers=True,
        patch_artist=True,
        medianprops=dict(color="#1f2328", lw=2),
        boxprops=dict(edgecolor="#8b949e"),
        whiskerprops=dict(color="#8b949e"),
        capprops=dict(color="#8b949e"),
        flierprops=dict(marker="o", markersize=3, markeredgecolor="none", alpha=0.7),
    )
    for box, flier, c in zip(bp["boxes"], bp["fliers"], colors):
        box.set_facecolor(matplotlib.colors.to_rgba(c, BOX_FILL_ALPHA))
        flier.set_markerfacecolor(c)
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
    models = sorted({r["model"] for r in runs})
    model_of = {r["variant"]: r["model"] for r in runs}
    colors = [MODEL_COLORS[models.index(model_of[v]) % len(MODEL_COLORS)] for v in variants]
    labels = [f"{v}\n{len(solved[v])}/{total[v]} solved ({100 * len(solved[v]) / total[v]:.0f}%)"
              for v in variants]

    boxplot(variants,
            [[r["n_turns"] for r in solved[v]] for v in variants],
            labels,
            colors,
            "turns to solve",
            "Turns per solved game",
            f"{args.out_prefix}_turns.png")
    boxplot(variants,
            [[r["n_tokens_out_total"] / 1000 for r in solved[v]] for v in variants],
            labels,
            colors,
            "total output tokens per game (thousands)",
            "Output tokens per solved game",
            f"{args.out_prefix}_tokens.png")

if __name__ == "__main__":
    main()
