"""Per-effort charts vs batch size for one model (mean over sessions).

  solved: % of puzzles solved
  tokens: output tokens per session
  tokens_per_puzzle: output tokens per session / batch size
  tradeoff: % solved vs tokens per puzzle (x is tokens, not batch size)

Usage: python3 scripts/plot_effort.py [--chart solved|tokens|tokens_per_puzzle|tradeoff|all] [--model M]
Writes out/effort_<chart>.png.
"""
import argparse
import collections
import glob
import json

import matplotlib
import matplotlib.transforms
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Ordinal blue ramp (light -> dark = low -> max effort).
COLORS = {"low": "#86b6ef", "high": "#2a78d6", "max": "#104281"}
N_INPUTS = 60  # endgames60: every batch size covers all 60 positions once
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e5e1"

METRICS = {
    "solved": dict(
        value=lambda r, b: 100.0 * r["n_solved"] / b,
        title="More reasoning effort delays the drop in accuracy",
        ylabel="Puzzles solved",
        ylim=(0, 104), yticks=range(0, 101, 20), fmt=lambda v, _: f"{v:.0f}%",
    ),
    "tokens": dict(
        value=lambda r, b: r["n_tokens_out"],
        title="Higher effort means a larger token budget per session",
        ylabel="Output tokens per session",
        ylim=(0, None), yticks=None, fmt=lambda v, _: f"{v / 1000:.0f}k",
    ),
    "tokens_per_puzzle": dict(
        value=lambda r, b: r["n_tokens_out"] / b,
        title="Tokens per puzzle shrink as the batch grows",
        ylabel="Output tokens per puzzle",
        ylim=(0, None), yticks=None, fmt=lambda v, _: f"{v / 1000:.0f}k",
    ),
}


def load(model, value):
    sessions = collections.defaultdict(list)  # (effort, bsz) -> [value]
    for f in glob.glob("logs/*.json"):
        with open(f) as fh:
            d = json.load(fh)
        if d["args"].get("model") != model:
            continue
        b = len(d["paths"])
        sessions[(d["args"]["reasoning_effort"], b)].append(value(d["result"], b))
    return sessions


def style(ax, model, title, n_series):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=20)
    ax.text(0, 1.02, f"{model} · mean over sessions", transform=ax.transAxes,
            fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncols=n_series, labelcolor=INK, handletextpad=0.2, columnspacing=1.0,
              borderaxespad=0.2)



def complete(sessions):
    """Drop sweeps still in progress: keep a batch size once all N_INPUTS puzzles ran."""
    return {k: v for k, v in sessions.items() if len(v) * k[1] >= N_INPUTS}


def mean(v):
    return sum(v) / len(v)


def plot(model, metric, out):
    m = METRICS[metric]
    sessions = load(model, m["value"])
    sessions = complete(sessions)
    efforts = [e for e in COLORS if any(k[0] == e for k in sessions)]
    sizes = sorted({k[1] for k in sessions})

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    for e in efforts:
        xs = [b for b in sizes if (e, b) in sessions]
        means = [sum(sessions[(e, b)]) / len(sessions[(e, b)]) for b in xs]
        ax.plot(xs, means, color=COLORS[e], lw=2, zorder=3)
        ax.scatter(xs, means, s=36, color=COLORS[e], edgecolors="white", linewidths=1.5,
                   zorder=4, label=e)
        ax.annotate(e, (xs[-1], means[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlim(sizes[0] / 1.15, sizes[-1] * 1.35)
    ax.set_ylim(*m["ylim"])
    if m["yticks"] is not None:
        ax.set_yticks(m["yticks"])
    ax.yaxis.set_major_formatter(FuncFormatter(m["fmt"]))
    ax.set_xlabel("Puzzles per session (batch size)", color=MUTED, fontsize=9)
    ax.set_ylabel(m["ylabel"], color=MUTED, fontsize=9)
    style(ax, model, m["title"], len(efforts))

    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


def label_points(fig, ax, points):
    """Batch size next to each point, at the first spot clear of other labels and dots."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # dot footprints in display space (36pt^2 marker ~ 6pt across, at 200 dpi)
    r = 4 * fig.dpi / 72
    dots = [ax.transData.transform((x, y)) for x, y, _ in points]
    placed = [matplotlib.transforms.Bbox([[px - r, py - r], [px + r, py + r]]) for px, py in dots]
    spots = [(5, -9, "left"), (-5, -9, "right"), (5, 4, "left"), (-5, 4, "right")]
    for x, y, b in sorted(points):
        # a near-touching dot below this one: label above, so labels don't swap
        px, py = ax.transData.transform((x, y))
        below = any(0 < py - qy < 4 * r and abs(px - qx) < 4 * r for qx, qy in dots)
        order = spots[2:] + spots[:2] if below else spots
        for i, (dx, dy, ha) in enumerate(order):
            t = ax.annotate(str(b), (x, y), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, fontsize=7, color=MUTED)
            box = t.get_window_extent(renderer)
            if not any(box.overlaps(o) for o in placed) or i == len(order) - 1:
                placed.append(box)
                break
            t.remove()


def plot_tradeoff(model, out):
    """% solved vs tokens per puzzle: one point per (effort, batch size)."""
    solved = complete(load(model, METRICS["solved"]["value"]))
    tpp = complete(load(model, METRICS["tokens_per_puzzle"]["value"]))
    efforts = [e for e in COLORS if any(k[0] == e for k in solved)]

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    points = []  # (x, y, batch size)
    for e in efforts:
        sizes = sorted(b for (e2, b) in solved if e2 == e)
        xs = [mean(tpp[(e, b)]) for b in sizes]
        ys = [mean(solved[(e, b)]) for b in sizes]
        ax.scatter(xs, ys, s=36, color=COLORS[e], edgecolors="white", linewidths=1.5,
                   zorder=4, label=e)
        points += zip(xs, ys, sizes)

    ax.set_xlim(0, None)
    ax.set_ylim(0, 104)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(FuncFormatter(METRICS["solved"]["fmt"]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlabel("Output tokens per puzzle (number next to a point = batch size)",
                  color=MUTED, fontsize=9)
    ax.set_ylabel("Puzzles solved", color=MUTED, fontsize=9)
    style(ax, model, "Accuracy tracks tokens per puzzle, whatever the effort", len(efforts))
    label_points(fig, ax, points)

    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    p.add_argument("--chart", choices=[*METRICS, "tradeoff", "all"], default="all")
    args = p.parse_args()
    for chart in [*METRICS, "tradeoff"] if args.chart == "all" else [args.chart]:
        out = f"out/effort_{chart}.png"
        if chart == "tradeoff":
            plot_tradeoff(args.model, out)
        else:
            plot(args.model, chart, out)


if __name__ == "__main__":
    main()
