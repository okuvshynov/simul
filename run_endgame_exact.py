#!/usr/bin/env python3
"""Endgame benchmark, exact continuation: the model takes over a game in progress
whose history is noisy, but every response to its own guesses is exact.

Positions come from data/endgames.jsonl (see make_endgames.py). The history is
given to the model as text; the game then continues live with fresh corruption
with NO corruption until solved or the turn budget runs out. Logs go to
endgame_logs/ and carry "continuation": "exact".

    python3 run_endgame_exact.py -m gpt-5.6-sol --reasoning-effort low --limit 20
    python3 run_endgame_exact.py --band forced --base-url http://localhost:8080/v1
"""
import argparse
import datetime
import httpx
import json
import openai
import os
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver  # noqa: E402

DATASET      = [str(d) for d in range(1000, 10000) if len(set(str(d))) == 4]
MAX_TURNS    = 6           # budget from the cut; with exact responses the solver needs 1-3
CONTINUATION = "exact"     # marker written to every log
API_TIMEOUT  = 7200
MAX_TOKENS   = 2 ** 16
MAX_ATTEMPTS = 3
RETRYABLE_STATUS = (408, 409, 429)

RULES = """
Let's play a game of Noisy Mastermind.

It is a variant of Mastermind game with imperfect communication channel.

# Rules

Codemaker comes up with a secret number.
The number has exactly four distinct digits 0..9.
Digit repetitions are not allowed in this variation - every digit is unique.
First digit cannot be 0, number must start with 1..9.

Examples of valid secret numbers: 1234, 1290, 9081

Examples of invalid secret numbers: 0123, 1111, 9912, 12345, 468

Example procedure to generate all valid secret numbers:

secrets = [str(d) for d in range(1000, 10000) if len(set(str(d))) == 4]

You are the codebreaker, your task is to guess the secret number.

At each turn you make a single guess. Your guess must be a potentially valid
secret with exact same restrictions - part of the secrets set.
Do not try to cheat by making a guess with repeated characters,
non-digits or shorter/longer guesses.

After every guess, you'll get two numbers as an output:
1. Number of digits you guessed correctly which are in the right position.
This number is often referred to as 'black'.
2. Number of digits you guessed correctly which are in the wrong position;
This number is often referred to as 'white'.

The final output would be a single string "black white".

Examples:
"0 0" - you guessed nothing
"0 4" - you guessed every digit, but it's completely wrong permutation
"4 0" - you solved the puzzle

Use the provided tool 'make_guess' to make a guess.
You MUST make EXACTLY ONE tool call per turn.

## Noisy channel (applies to the game history only)

The guesses you will see in the history below were made over a noisy channel.
On that channel, if a guess was exactly the secret, the response was always
"4 0". Otherwise each digit of the guess was corrupted before comparison,
independently, with probability p = {p_corruption}.

A corrupted digit matches nothing - it is not equal to any digit. So on the
noisy channel corruption could lower the number of matches reported, but never
raise it: every response in the history is a lower bound on the true number
of black and white matches for that guess.

### Examples for the noisy channel

secret = 1234, guess = 1235, corrupted to "1?35": the response was "2 0"
- digit "2" was corrupted and ignored.

secret = 1234, guess = 1236, corrupted to "1??6": the response was "1 0".

secret = 1234, guess = 1238, no digit corrupted: the response was "3 0".

secret = 1234, guess = 1234: exact match, the response was "4 0".

# Scoring

The number of guesses you need to solve the puzzle will be recorded.
If you fail to find the solution within {max_turns} guesses, the puzzle is
marked as unsolved. Your priorities are (in order):

1. Solve as many puzzles as possible.
2. Minimize the average number of guesses for solved ones.

If invalid guess is encountered, for example:

- number starting with 0
- repeated digits
- non-digits
- guess with number of digits other than 4

the output will be "invalid guess". The turn is lost and still counts
towards the limit.

# Game in progress

You are taking over a game that is already in progress. Another player made
the guesses below over the NOISY channel and received these responses:

{history}

The channel has now been repaired. From your first guess on, every response
you receive is EXACT: nothing is corrupted, and the response is the true
number of black and white matches for your guess.

So you have two kinds of evidence:
- the history above: noisy, each response is only a lower bound;
- responses to your own guesses: exact.

The guesses in the history do not count towards your limit of {max_turns} guesses.
Your turn, make the next guess.
"""

TOOLS = [{
    "type": "function",
    "name": "make_guess",
    "description": "Make a guess. The output is a string with two numbers: 'black white'",
    "parameters": {
        "type": "object",
        "properties": {"guess": {"type": "string", "description": "Four digit guess for the game."}},
        "required": ["guess"],
    },
}]


def build_prompt(pos):
    lines = [f'{i + 1}. guess {g} -> response "{b} {w}"' for i, (g, b, w) in enumerate(pos["history"])]
    return RULES.format(p_corruption=pos["p_corruption"], max_turns=MAX_TURNS, history="\n".join(lines))


def score_guess(guess, secret):
    """Exact response: no corruption in the continuation."""
    black = sum(a == b for a, b in zip(guess, secret))
    white = len(set(guess).intersection(secret)) - black
    return f"{black} {white}"


class ApiFailure(Exception):
    def __init__(self, err, attempts):
        super().__init__(str(err))
        self.attempts = attempts


TERMINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")

def create_response(client, **kwargs):
    """Streamed request with our own retries; see run.py for the reasoning."""
    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            final = None
            with client.responses.create(stream=True, **kwargs) as events:
                for ev in events:
                    if ev.type in TERMINAL_EVENTS:
                        final = ev.response
            if final is not None:
                return final, attempt
            last_err = RuntimeError("stream ended without a terminal event")
        except (openai.APIError, httpx.HTTPError) as e:
            last_err = e
            if isinstance(e, openai.APIStatusError) and e.status_code < 500 and e.status_code not in RETRYABLE_STATUS:
                print(f"W: client error, not retrying: {e}")
                raise ApiFailure(e, attempt)
        print(f"W: attempt {attempt}/{MAX_ATTEMPTS} failed: {last_err}")
        if attempt < MAX_ATTEMPTS:
            time.sleep(5 * attempt)
    raise ApiFailure(last_err, MAX_ATTEMPTS)


def posterior_at(pos):
    """Solver's view of the position, used to score the model's first guess."""
    hist = [(g, b, w) for g, b, w in pos["history"]]
    return solver.posterior(DATASET, hist, pos["p_corruption"])


def play(pos, model, reasoning_effort, tool_choice, client):
    secret = pos["secret"]
    input_list = [{"role": "user", "content": build_prompt(pos)}]
    extra = {"tool_choice": "required"} if tool_choice == "required" else {}
    trace = []
    first_guess = None

    for turn in range(MAX_TURNS):
        try:
            response, attempts = create_response(
                client, input=input_list, model=model, tools=TOOLS,
                reasoning={"effort": reasoning_effort}, max_output_tokens=MAX_TOKENS,
                parallel_tool_calls=False, **extra,
            )
        except ApiFailure as e:
            print(f"W: giving up on turn {turn + 1}: {e}")
            trace.append({"input_tokens": 0, "output_tokens": 0, "n_calls": 0,
                          "attempts": e.attempts, "status": "err_api"})
            return trace, first_guess

        trace.append({
            "input_tokens":  response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "status": "",
            "n_calls": 0,
            "attempts": attempts,
        })
        input_list += response.output

        if response.status != "completed":
            trace[-1]["status"] = "err_response"
            print(f"W: response status {response.status}")
            return trace, first_guess

        calls = [it for it in response.output if it.type == "function_call" and it.name == "make_guess"]
        trace[-1]["n_calls"] = len(calls)
        if len(calls) != 1:
            trace[-1]["status"] = "err_n_calls"
            print(f"W: expected one guess per turn, got {len(calls)}")
            return trace, first_guess

        try:
            guess = json.loads(calls[0].arguments)["guess"]
        except (json.JSONDecodeError, TypeError, KeyError):
            guess = calls[0].arguments
        trace[-1]["guess"] = guess

        if turn == 0:
            post = posterior_at(pos)
            best = max(post, key=post.get)
            first_guess = {"guess": guess, "p_guess": post.get(guess, 0.0), "best": best, "p_best": post[best]}

        if guess not in DATASET:
            res = "invalid guess"
            print(f"W: #{turn + 1} invalid guess '{guess}' | out_tokens = {trace[-1]['output_tokens']}")
        else:
            res = score_guess(guess, secret)
            print(f"I: #{turn + 1} g({guess}, {secret}) = {res} [exact] | out_tokens = {trace[-1]['output_tokens']}")

        trace[-1]["res"] = res
        if res == "4 0":
            trace[-1]["status"] = "solved"
            return trace, first_guess

        input_list.append({"type": "function_call_output", "call_id": calls[0].call_id, "output": res})

    trace[-1]["status"] = "unsolved"
    return trace, first_guess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/endgames.jsonl")
    ap.add_argument("--band", help="only positions from this band (forced, eff3, ...)")
    ap.add_argument("--limit", type=int, help="stop after this many positions")
    ap.add_argument("--offset", type=int, default=0, help="skip this many positions first")
    ap.add_argument("--model", "-m")
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--tool-choice", choices=["required", "auto"], default="required")
    ap.add_argument("--base-url")
    ap.add_argument("--api-key")
    ap.add_argument("--out-dir", default="endgame_logs")
    args = ap.parse_args()

    if args.base_url is not None:
        print(f"I: base-url override: {args.base_url}")
        api_key = args.api_key or "sk-no-key"
        client = openai.OpenAI(base_url=args.base_url, api_key=api_key, timeout=API_TIMEOUT, max_retries=0)
    else:
        client = openai.OpenAI(api_key=args.api_key, timeout=API_TIMEOUT, max_retries=0)

    if args.model is None:
        models = client.models.list().data
        if len(models) != 1:
            sys.exit("E: no model specified; pass -m (endpoint lists %d models)" % len(models))
        model = models[0].id
    else:
        model = args.model
    print(f"I: model: {model}")

    positions = [json.loads(line) for line in open(args.data)]
    if args.band:
        positions = [p for p in positions if p["band"] == args.band]
    positions = positions[args.offset:]
    if args.limit is not None:
        positions = positions[:args.limit]
    print(f"I: {len(positions)} positions")

    os.makedirs(args.out_dir, exist_ok=True)
    for n, pos in enumerate(positions):
        dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        st = pos["stats"]
        print(f"I: position {n + 1}/{len(positions)} id={pos['id'][:8]} band={pos['band']} "
              f"secret={pos['secret']} support={st['support']} eff_n={st['eff_n']} solver_turns={st['solver_turns']}")
        trace, first_guess = play(pos, model, args.reasoning_effort, args.tool_choice, client)
        content = {
            "status": trace[-1]["status"],
            "model": model,
            "reasoning_effort": args.reasoning_effort,
            "position": pos,
            "continuation": CONTINUATION,
            "max_turns": MAX_TURNS,
            "turns": len(trace),
            "total_gen_tokens": sum(t["output_tokens"] for t in trace),
            "invalid_guesses": sum(t.get("res") == "invalid guess" for t in trace),
            "first_guess": first_guess,
            "trace": trace,
            "args": {"data": args.data, "band": args.band, "offset": args.offset, "limit": args.limit,
                     "tool_choice": args.tool_choice},
        }
        tag = secrets.token_hex(3)
        with open(f"{args.out_dir}/{dt}-{pos['id'][:8]}-{tag}.json", "x") as fw:
            fw.write(json.dumps(content))


if __name__ == "__main__":
    main()
