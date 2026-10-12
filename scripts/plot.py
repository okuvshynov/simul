import argparse
import glob
import json
from collections import defaultdict

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_data(cb_filter=None):
    turns = defaultdict(list)
    solved = defaultdict(list)
    turns_solved = defaultdict(list)

    for path in glob.glob("logs/*.json"):
        with open(path) as f:
            r = json.load(f)
        key = (r["model"], r["args"]["reasoning_effort"], r["args"]["n_games"])

        if cb_filter:
            if not cb_filter(key):
                continue

        n_turns = r["n_turns"]
        n_games = r["args"]["n_games"]
        n_solved = len(r["solved"])
        n_turns_solved = sum(1 for t in r["trace"] if "codemaker" in t and t["codemaker"] in r["solved"])

        # it's ok to normalize by n_games here because we are not aggregating 
        # across n_games. Each data point in a group will have same denominator
        turns[key].append(n_turns / n_games)
        solved[key].append(n_solved / n_games)
        turns_solved[key].append(n_turns_solved / n_games)

    return turns, solved, turns_solved


def mean_diff_ci(x, base, conf=0.95):
    """Difference in means (x - base) with Welch's t confidence interval."""
    x, base = np.asarray(x), np.asarray(base)
    vx, vb = x.var(ddof=1) / len(x), base.var(ddof=1) / len(base)
    se = np.sqrt(vx + vb)
    df = (vx + vb) ** 2 / (vx**2 / (len(x) - 1) + vb**2 / (len(base) - 1))
    half = stats.t.ppf((1 + conf) / 2, df) * se
    return x.mean() - base.mean(), half

def plot_diffs(groups, baseline):
    base = groups[baseline]
    keys = sorted(k for k in groups if k != baseline)

    diffs, errs = zip(*(mean_diff_ci(groups[k], base) for k in keys))
    labels = [f"n_games = {n_games}" for model, effort, n_games in keys]
    y = np.arange(len(keys))

    plt.errorbar(diffs, y, xerr=errs, fmt="o", capsize=4)
    plt.axvline(0, color="gray", linestyle="--")
    plt.yticks(y, labels)
    plt.xlabel(f"Δ mean n_turns vs n_games=1 (95% CI)")
    plt.tight_layout()
    plt.show()

def plot_boxes(groups):
    keys = sorted(groups)
    data = [groups[k] for k in keys]
    labels = [f"{model}-{effort}\n{n_games}" for model, effort, n_games in keys]

    plt.boxplot(data, tick_labels=labels, vert=False)
    plt.xlabel("n_turns")
    plt.show()

def main():
    f = lambda key : "sol" in key[0]
    turns, solved, ts = load_data(f)
    plot_boxes(turns)
    #plot_boxes(solved)
    #plot_diffs(ts, ("deepseek-flash", "low", 1))
    plot_diffs(turns, ("gpt-6.1-sol", "max", 20))
    #plot_diffs(solved, ("gpt-5.6-terra", "max", 1))
    #plot_diffs(ts, ("gpt-5.6-terra", "max", 1))
    #plot_diffs(turns_solved, ("gpt-5.6-terra", "max", 1))
    #plot_diffs(turns_solved, ("deepseek-flash", "low", 1))

if __name__ == "__main__":
    main()
