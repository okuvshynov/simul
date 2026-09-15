import argparse
import json
import os


def load_run(path):
    with open(path) as f:
        return json.load(f)


def summarize_dir(dirpath):
    runs = []
    for name in sorted(os.listdir(dirpath)):
        full = os.path.join(dirpath, name)
        if not os.path.isfile(full):
            continue
        try:
            runs.append(load_run(full))
        except (json.JSONDecodeError, OSError):
            print(f"warning: skipping unreadable file {full}")
    if not runs:
        return None

    n = len(runs)
    successes = sum(1 for r in runs if r["success"])
    steps = [len(r["usage"]) for r in runs]
    completion = [sum(u["completion_tokens"] for u in r["usage"]) for r in runs]
    prompt = [sum(u["prompt_tokens"] for u in r["usage"]) for r in runs]

    return {
        "name": os.path.basename(os.path.normpath(dirpath)),
        "samples": n,
        "successes": successes,
        "avg_steps": sum(steps) / n,
        "avg_completion": sum(completion) / n,
        "avg_completion_per_step": sum(completion) / sum(steps),
        "avg_prompt": sum(prompt) / n,
    }


def main():
    parser = argparse.ArgumentParser("Summarize Bulls & Cows benchmark logs")
    parser.add_argument("--logs", default="logs", help="directory containing one subdirectory per run")
    args = parser.parse_args()

    rows = []
    for name in sorted(os.listdir(args.logs)):
        full = os.path.join(args.logs, name)
        if os.path.isdir(full):
            row = summarize_dir(full)
            if row:
                rows.append(row)

    if not rows:
        print(f"no run directories found under {args.logs}")
        return

    header = ["run", "success", "avg steps", "avg completion tok", "tok / step", "avg prompt tok"]
    table = [
        [
            r["name"],
            f"{r['successes']}/{r['samples']}",
            f"{r['avg_steps']:.1f}",
            f"{r['avg_completion']:.0f}",
            f"{r['avg_completion_per_step']:.0f}",
            f"{r['avg_prompt']:.0f}",
        ]
        for r in rows
    ]

    widths = [max(len(str(x)) for x in col) for col in zip(header, *table)]

    def fmt(row):
        cells = []
        for i, cell in enumerate(row):
            cells.append(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]))
        return "| " + " | ".join(cells) + " |"

    print(fmt(header))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in table:
        print(fmt(row))


if __name__ == "__main__":
    main()
