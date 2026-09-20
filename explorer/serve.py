#!/usr/bin/env python3
"""Tiny web explorer for logs/. Run from anywhere: python3 explorer/serve.py [port]"""

import json
import math
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
ENDGAME_LOGS = ROOT / "endgame_logs"

sys.path.insert(0, str(ROOT))
import solver  # noqa: E402

DATASET = [str(d) for d in range(1000, 10000) if len(set(str(d))) == 4]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path == "/":
            self.path = "/explorer/index.html"
        elif self.path == "/api/logs":
            return self.send_json(list_logs(LOGS))
        elif self.path == "/api/endgame/logs":
            return self.send_json(list_logs(ENDGAME_LOGS))
        elif self.path.startswith("/api/replay/"):
            return self.send_replay(LOGS, self.path[len("/api/replay/"):])
        elif self.path.startswith("/api/endgame/replay/"):
            return self.send_replay(ENDGAME_LOGS, self.path[len("/api/endgame/replay/"):])
        return super().do_GET()

    def send_replay(self, directory, name):
        name = unquote(name)
        path = directory / name
        if "/" in name or not path.is_file():
            return self.send_json({"error": "no such log"})
        return self.send_json(replay(json.loads(path.read_text())))

    def end_headers(self):
        # logs change while runs are in progress; never let the browser cache them
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet


def replay(d):
    """Run solver.py over a game, one turn at a time, and report per-turn
    solver stats. Weights are kept unnormalised and updated incrementally,
    so the whole game costs O(turns * |DATASET|) likelihood evaluations.

    Endgame logs carry the position they started from; its history is replayed
    first and those rows are flagged `given`, so row i always describes step i
    of (given history + model trace)."""
    if "position" in d:
        pos = d["position"]
        p, secret = pos["p_corruption"], pos["secret"]
        steps = [{"guess": g, "res": f"{b} {w}", "given": True} for g, b, w in pos["history"]] + d["trace"]
    else:
        p, secret = d["p_corruption"], d["secret"]
        steps = d["trace"]
    weights = {s: 1.0 for s in DATASET}
    rows = []
    for i, t in enumerate(steps):
        row = {"turn": i + 1, "given": bool(t.get("given"))}
        res = t.get("res")
        if res is None or res == "invalid guess":
            rows.append(row)
            continue
        g = t["guess"]
        Z = sum(weights.values())
        best = max(weights, key=weights.get)
        row.update(p_guess=weights.get(g, 0.0) / Z, best=best, p_best=weights[best] / Z)
        b, w = map(int, res.split())
        if (b, w) == (4, 0):
            rows.append(row)
            break
        new = {}
        for s, wt in weights.items():
            if s == g:
                continue
            l = solver.likelihood(b, w, *solver.score(g, s), p)
            if l > 0.0:
                new[s] = wt * l
        Z = sum(new.values())
        if Z == 0.0:
            row["error"] = "history inconsistent with the rules"
            rows.append(row)
            break
        weights = {s: v / Z for s, v in new.items()}   # renormalise to avoid underflow
        H = -sum(v * math.log(v) for v in weights.values())
        pt = weights.get(secret, 0.0)
        # many candidates tie exactly; a tolerance keeps rank stable under rounding
        row.update(support=len(weights), eff_n=math.exp(H), p_truth=pt,
                   rank=1 + sum(1 for v in weights.values() if v > pt * (1 + 1e-9)))
        rows.append(row)
    return rows


def list_logs(directory):
    """All logs in a directory without their trace, plus the file name to link to.
    Endgame logs also drop the position's history, which the list pages don't need."""
    out = []
    for p in sorted(directory.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue  # partially written file, skip for now
        d.pop("trace", None)
        if "position" in d:
            d["position"] = {k: v for k, v in d["position"].items() if k != "history"}
        d["file"] = p.name
        out.append(d)
    return out


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"serving {LOGS} at http://localhost:{port}/")
    ThreadingHTTPServer(("", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
