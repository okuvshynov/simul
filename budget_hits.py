import argparse
import glob
import json
import re

# made by claude

# Qwen-specific: how often a single step hit the reasoning budget.
# Budget comes from the log note (rb1024 -> 1024 tokens). A step counts as a hit
# when its output tokens exceed the budget; the tool call adds a few tokens on
# top of reasoning, so this can slightly overcount steps that ended just short.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs/*.json")
    args = parser.parse_args()

    stats = {}
    for p in glob.glob(args.logs):
        with open(p) as f:
            r = json.load(f)
        m = re.fullmatch(r"rb(\d+)", r.get("note") or "")
        if not r["model"].startswith("qwen") or m is None:
            continue
        budget = int(m.group(1))
        s = stats.setdefault(budget, {"steps": 0, "hits": 0, "games": 0, "games_hit": 0})
        hits = sum(step["n_tokens_out"] > budget for step in r["trace"])
        s["steps"]     += len(r["trace"])
        s["hits"]      += hits
        s["games"]     += 1
        s["games_hit"] += hits > 0

    print("| budget | steps over | % steps | games with hit | % games |")
    print("|---:|---:|---:|---:|---:|")
    for budget in sorted(stats):
        s = stats[budget]
        print(f"| {budget} | {s['hits']}/{s['steps']} | {100 * s['hits'] / s['steps']:.1f}%"
              f" | {s['games_hit']}/{s['games']} | {100 * s['games_hit'] / s['games']:.1f}% |")

if __name__ == "__main__":
    main()
