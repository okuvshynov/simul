#!/usr/bin/env python3
"""Extract endgame positions from game logs.

A position is what the solver sees plus the answer key: the secret, p_corruption
and the history of valid guesses with their (corrupted) responses. Each game is
walked turn by turn, and a position is emitted the first time the game enters
each requested difficulty band. Labels come from solver.py.

    python3 make_endgames.py                      # logs/ + old_logs/ -> data/endgames.jsonl
    python3 make_endgames.py --bands forced,eff3,eff10 --logs logs
"""
import argparse
import glob
import hashlib
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver  # noqa: E402

DATASET = [str(d) for d in range(1000, 10000) if len(set(str(d))) == 4]

# band name -> predicate on the solver stats after the cut
BANDS = {
    "forced": lambda st: st["support"] == 1,
    "eff3":   lambda st: st["support"] > 1 and st["eff_n"] <= 3,
    "eff10":  lambda st: 3 < st["eff_n"] <= 10,
    "eff30":  lambda st: 10 < st["eff_n"] <= 30,
    "eff100": lambda st: 30 < st["eff_n"] <= 100,
}


def update(post, g, b, w, p):
    """Posterior after one more observation. Returns {} if inconsistent."""
    new = {}
    for s, wt in post.items():
        if s == g:
            continue
        l = solver.likelihood(b, w, *solver.score(g, s), p)
        if l > 0.0:
            new[s] = wt * l
    Z = sum(new.values())
    return {s: v / Z for s, v in new.items()} if Z > 0.0 else {}


def stats(post, secret):
    H = -sum(v * math.log(v) for v in post.values())
    best = max(post, key=post.get)
    pt = post.get(secret, 0.0)
    return {
        "support": len(post),
        "eff_n": round(math.exp(H), 3),
        "best": best,
        "p_best": round(post[best], 4),
        "p_truth": round(pt, 4),
        "rank": 1 + sum(1 for v in post.values() if v > pt * (1 + 1e-9)),
    }


def corrupted_score(guess, secret, p, rng):
    """Same rule as run.py: exact match is never corrupted."""
    if guess == secret:
        return 4, 0
    g = "".join("?" if rng.random() < p else c for c in guess)
    b = sum(a == c for a, c in zip(g, secret))
    return b, len(set(g) & set(secret)) - b


def solver_turns(post, secret, p, seed, n=50, cap=30):
    """Mean turns for the greedy solver (always guess the top candidate) to
    finish from this position, over n simulated continuations."""
    rng = random.Random(seed)
    total = 0
    for _ in range(n):
        cur, turns = post, 0
        while turns < cap:
            g = max(cur, key=cur.get)
            turns += 1
            b, w = corrupted_score(g, secret, p, rng)
            if (b, w) == (4, 0):
                break
            cur = update(cur, g, b, w, p)
        total += turns
    return round(total / n, 2)


def positions_from_game(d, source_file, bands):
    p, secret = d["p_corruption"], d["secret"]
    post = {s: 1.0 / len(DATASET) for s in DATASET}
    history = []
    seen = set()
    for i, t in enumerate(d["trace"]):
        res = t.get("res")
        if res is None or res == "invalid guess":
            continue
        b, w = map(int, res.split())
        if (b, w) == (4, 0):
            break
        history.append([t["guess"], b, w])
        post = update(post, t["guess"], b, w, p)
        if not post:
            break
        st = stats(post, secret)
        for name in bands:
            if name in seen or not BANDS[name](st):
                continue
            seen.add(name)
            pid = hashlib.md5(json.dumps([secret, history]).encode()).hexdigest()
            st["solver_turns"] = solver_turns(post, secret, p, seed=int(pid[:8], 16))
            yield {
                "id": pid,
                "secret": secret,
                "p_corruption": p,
                "history": [list(h) for h in history],
                "band": name,
                "stats": st,
                "source": {
                    "file": source_file,
                    "model": d.get("model"),
                    "reasoning_effort": d.get("reasoning_effort"),
                    "turn": i + 1,
                },
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", default=["logs", "old_logs"], help="directories with game logs")
    ap.add_argument("--bands", default="forced,eff3", help="comma-separated subset of " + ",".join(BANDS))
    ap.add_argument("--out", default="data/endgames.jsonl")
    ap.add_argument("--rebuild", action="store_true",
                    help="start from scratch instead of appending to --out. Changes position order, "
                         "so --limit/--offset runs made against the old file no longer line up.")
    args = ap.parse_args()

    bands = args.bands.split(",")
    unknown = [b for b in bands if b not in BANDS]
    if unknown:
        sys.exit(f"E: unknown band(s) {unknown}; choose from {list(BANDS)}")

    # Append-only by default: existing positions keep their order (and ids are
    # content hashes), so earlier --limit/--offset runs stay paired.
    out = {}
    if not args.rebuild and Path(args.out).exists():
        for line in open(args.out):
            pos = json.loads(line)
            out[pos["id"]] = pos
    kept = len(out)

    files = sorted(f for d in args.logs for f in glob.glob(f"{d}/*.json"))
    per_band, games = {}, 0
    for f in files:
        d = json.load(open(f))
        if not d.get("trace"):
            continue
        games += 1
        for pos in positions_from_game(d, f, bands):
            if pos["id"] in out:
                continue
            out[pos["id"]] = pos
            per_band[pos["band"]] = per_band.get(pos["band"], 0) + 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fw:
        for pos in out.values():
            fw.write(json.dumps(pos) + "\n")

    print(f"I: scanned {games} games from {len(files)} files")
    print(f"I: kept {kept} existing positions")
    for b in bands:
        print(f"I: {b:8} {per_band.get(b, 0):5} new positions")
    print(f"I: wrote {len(out)} positions to {args.out}")


if __name__ == "__main__":
    main()
