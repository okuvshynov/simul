import argparse
import glob
import json
from collections import defaultdict

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_groups():
    groups = defaultdict(list)
    for path in glob.glob("logs/*.json"):
        with open(path) as f:
            r = json.load(f)
        key = (r["model"], r["args"]["reasoning_effort"], r["args"]["n_games"])
        groups[key].append(r["n_turns"] / r["args"]["n_games"])
    return groups


def mean_diff_ci(x, base, conf=0.95):
    """Difference in means (x - base) with Welch's t confidence interval."""
    x, base = np.asarray(x), np.asarray(base)
    vx, vb = x.var(ddof=1) / len(x), base.var(ddof=1) / len(base)
    se = np.sqrt(vx + vb)
    df = (vx + vb) ** 2 / (vx**2 / (len(x) - 1) + vb**2 / (len(base) - 1))
    half = stats.t.ppf((1 + conf) / 2, df) * se
    return x.mean() - base.mean(), half


def plot_diffs(groups, baseline):
    groups = load_groups()
    base = groups[baseline]
    keys = sorted(k for k in groups if k != baseline)

    diffs, errs = zip(*(mean_diff_ci(groups[k], base) for k in keys))
    labels = [f"{model}-{effort}\n{n_games}" for model, effort, n_games in keys]
    y = np.arange(len(keys))

    plt.errorbar(diffs, y, xerr=errs, fmt="o", capsize=4)
    plt.axvline(0, color="gray", linestyle="--")
    plt.yticks(y, labels)
    plt.xlabel(f"Δ mean n_turns vs {'-'.join(map(str, baseline))} (95% CI)")
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
    groups = load_groups()
    plot_boxes(groups)
    plot_diffs(groups, ("deepseek-flash", "low", 1))

if __name__ == "__main__":
    main()
