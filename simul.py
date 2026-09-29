import argparse
import datetime
import httpx
import json
import openai
import os
from pprint import pprint
import random
import secrets
import time

import noisy_mm

CODEMAKERS = [
    ("Red",        "#E6194B"),
    ("Green",      "#3CB44B"),
    ("Yellow",     "#FFE119"),
    ("Blue",       "#4363D8"),
    ("Orange",     "#F58231"),
    ("Purple",     "#911EB4"),
    ("Cyan",       "#42D4F4"),
    ("Magenta",    "#F032E6"),
    ("Lime",       "#BFEF45"),
    ("Pink",       "#FABED4"),
    ("Teal",       "#469990"),
    ("Lavender",   "#DCBEFF"),
    ("Brown",      "#9A6324"),
    ("Maroon",     "#800000"),
    ("Olive",      "#808000"),
    ("Mint",       "#AAFFC3"),
    ("Navy",       "#2C376F"),
    ("Sky",        "#6FA6FF"),
    ("Petrol",     "#16646F"),
    ("Aqua",       "#00FFF4"),
    ("Forest",     "#004E2C"),
    ("Kelly",      "#0B7A00"),
    ("Khaki",      "#B1B185"),
    ("Umber",      "#4E4300"),
    ("Mustard",    "#D39B00"),
    ("Vermilion",  "#D33700"),
    ("Peach",      "#FFD3B1"),
    ("Clay",       "#B17A6F"),
    ("Flamingo",   "#F46F9B"),
    ("Raspberry",  "#BC006F"),
    ("Plum",       "#64374E"),
    ("Mauve",      "#907AA6"),
]

# default is 10 min. Bump to 2 hours for local runs
API_TIMEOUT  = 7200

# API calls per turn before giving up on the sample.
N_ATTEMPTS_MAX = 3

def format_prompt(n_turns_max, p_corr, n_games, codemakers, next_codemaker):
    return f"""
Let's play a game of Noisy Mastermind.

It is a variant of Mastermind game with imperfect communication channel.
You are a Grandmaster who plays simul session with multiple codemakers.

# Rules

Each codemaker comes up with a secret code.
The code is a number with exactly four distinct digits 0..9.
Digit repetitions are not allowed in this variation - every digit is unique in a given code.
First digit cannot be 0, code must start with 1..9.

Examples of valid secret codes: 1234, 1290, 9081

Examples of invalid secret codes: 0123, 1111, 9912, 12345, 468

Example procedure to generate all valid secret codes:

codes = [str(d) for d in range(1000, 10000) if len(set(str(d))) == 4]

You are the codebreaker, your task is to guess each secret code.

At each turn you make a single guess to one codemaker. 
Your guess must be a potentially valid code with exact same restrictions.
Do not try to cheat by making a guess with repeated characters, 
non-digits or shorter/longer guesses. Such turns will be wasted.

You play against multiple codemakers simultaneously, and turn order is unknown.

At every turn, you will be told which codemaker to query next.
You MUST query the designated next codemaker, otherwise your turn will be wasted.

After every guess, you'll get two numbers and next codemaker to query:
1. Number of digits you guessed correctly which are in the right position.
This number is often referred to as 'black'.
2. Number of digits you guessed correctly which are in the wrong position;
This number is often referred to as 'white'.
3. Nickname of the next codemaker to query.

Examples of "black white":
"0 0" - you guessed nothing
"0 4" - you guessed every digit, but it's completely wrong permutation
"4 0" - you solved the puzzle

Use the provided tool 'make_guess' to make a guess.
You MUST make EXACTLY ONE tool call per turn.

## Communication corruption.

If you guessed the secret number, you will always get "4 0". Codemaker will
not make any mistakes for exact match.

If not, each of the digits in your guess might be corrupted before comparison.
Corruption is independent for each digit and happens with probability
p = {p_corr}.

The response you get will be computed after corruption.
Corrupted digit is not equal to any digit and will not match anything.
Thus, corruption can decrease the number of matches you would get,
but never increase.

# Examples

## Example 1:

secret code = 1234
your guess  = 1235

Let's say one digit got corrupted and after corruption guess becomes "1?35"

the output you get is "2 0" - digit "2" was corrupted and ignored.

## Example 2:

secret code = 1234
your guess  = 1234

You guessed correctly, so there was no chance of corruption.
The output you get is "4 0".

## Example 3:

secret code = 1234
your guess  = 1236

after corruption, guess = 1??6

Two digits were corrupted, you'll get "1 0".

## Example 4:

secret code = 1234
your guess  = 1238

after corruption, guess = 1238 - no corruption happened in this case.

the output you get is "3 0".

# Scoring

For every game the number of guesses to solve each problem will be recorded.
You have {n_turns_max} guesses total; if you guessed only some of the puzzles, you'll get partial score. 

Your goals are:
1. Solve as many puzzles as possible.
2. Minimize the average number of guesses for solved puzzles.

If invalid guess is encountered, for example:

- number starting with 0
- repeated digits
- non-digits
- guess with number of digits other than 4
- guess directed to a wrong codemaker
- not one tool call

the output will be a string with error description. 
The turn is lost and still counts towards the {n_turns_max} limit.

You play a session with {n_games} games against codemakers: {codemakers}.
First codemaker to query is {next_codemaker}.
"""

TOOL_DESCRIPTION = """
You make a guess for one codemaker.
You will get the following output: "black white next_codemaker".
On your next tool call you MUST query the next_codemaker.
If you try to query someone else, the turn will be wasted.
"""

TOOLS = [{
    "type" : "function",
    "name" : "make_guess",
    "description" : TOOL_DESCRIPTION,
    "parameters" : {
        "type" : "object",
        "properties" : {
            "guess": {
                "type" : "string",
                "description" : "Four digit guess for the game."
            },
            "codemaker" : {
                "type" : "string",
                "description" : "Codemaker to ask. Make sure to use the right one!"
            }
        },
        "required" : ["guess", "codemaker"]
    }
}]

class ApiFailure(Exception):
    """Gave up on one turn: retries exhausted, or a client error that retrying can't fix."""
    def __init__(self, err, attempts):
        super().__init__(str(err))
        self.attempts = attempts

# We stream to keep connection alive; benchmark itself doesn't care, so
# we accumulate full response.
# 4xx responses (other than these) mean the request itself is wrong; retrying is pointless.
RETRYABLE_STATUS = (408, 409, 429)
TERMINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")

def create_response(client, **kwargs):
    last_err = None
    for attempt in range(1, N_ATTEMPTS_MAX + 1):
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
        print(f"W: attempt {attempt}/{N_ATTEMPTS_MAX} failed: {last_err}")
        if attempt < N_ATTEMPTS_MAX:
            time.sleep(5 * attempt)
    raise ApiFailure(last_err, N_ATTEMPTS_MAX)

def save_error_response(response, turn):
    os.makedirs("logs/errors", exist_ok=True)
    dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    path = f"logs/errors/{dt}-t{turn + 1}-{secrets.token_hex(3)}.json"
    with open(path, "x") as fw:
        fw.write(response.model_dump_json())
    print(f"W: saved full response to {path}")
    return path

def run(client, secret_codes, args):
    n_turns_max = args.n_turns_max * args.n_games

    codemakers = [name for name, _ in CODEMAKERS[:args.n_games]]
    prompt = format_prompt(
        n_turns_max=n_turns_max,
        p_corr=args.p_corr,
        n_games=args.n_games,
        codemakers=", ".join(codemakers),
        next_codemaker=codemakers[0]
    )

    input_list = [{"role": "user", "content": prompt}]
    trace = []

    # "required" + parallel_tool_calls=False for one call per turn. 
    # DeepSeek rejects it in thinking mode, so "auto" means 'default'.
    extra = {"tool_choice": "required"} if args.tool_choice == "required" else {}

    solved = []
    next_codemaker = codemakers[0]
    codes_by_name = {name: code for name, code in zip(codemakers, secret_codes)}

    for turn in range(n_turns_max):
        try:
            response, attempts = create_response(
                client,
                input=input_list,
                model=args.model,
                tools=TOOLS,
                reasoning={"effort" : args.reasoning_effort},
                max_output_tokens=args.n_tokens_max,
                parallel_tool_calls=False,
                **extra,
            )
        except ApiFailure as e:
            print(f"W: giving up on turn {turn + 1}: {e}")
            trace.append({
                "n_tokens_in"  : 0,
                "n_tokens_out" : 0,
                "n_attempts"   : e.attempts,
                "status"       : {"error" : "api_failure"}
            })
            return solved, trace

        trace.append({
            "n_tokens_in"  : response.usage.input_tokens,
            "n_tokens_out" : response.usage.output_tokens,
            "status"       : {},
            "n_attempts"   : attempts,
            "warnings"     : []
        })
        input_list += response.output

        if response.status != "completed":
            trace[-1]["status"] = { "error" : "response_status"}
            print(f"W: response error, possibly hit {args.n_tokens_max}.")
            trace[-1]["error_log"] = save_error_response(response, turn)
            return solved, trace

        # first, check that we have exactly one guess tool call per instructions
        calls = [
            item for item in response.output
            if item.type == "function_call" and item.name == "make_guess"
        ]

        if len(calls) != 1:
            trace[-1]["warnings"].append(f"n_tool_calls|{len(calls)}")
            res = f"Expected one guess tool call per turn, got {len(calls)}"
            print(f"W: {res}")
        else:
            try:
                call_args = json.loads(calls[0].arguments)
                guess = call_args["guess"]
                codemaker = call_args["codemaker"]
            except (json.JSONDecodeError, TypeError, KeyError):
                # this way we'll keep invalid guess in the logs for inspection
                guess = calls[0].arguments
                codemaker = "unknown"

            trace[-1]["guess"] = guess
            trace[-1]["codemaker"] = codemaker

            if guess not in noisy_mm.DATASET:
                # the turn is lost, but the game goes on
                res = "invalid guess"
                trace[-1]["warnings"].append(f"invalid_guess|{guess}")
                 
                print(f"W: #{(turn + 1):3} {codemaker:10} {guess} : {res}"
                      f" | tokens: in={trace[-1]['n_tokens_in']}, out={trace[-1]['n_tokens_out']}")
            elif codemaker != next_codemaker:
                # the turn is lost, but the game goes on
                trace[-1]["warnings"].append(f"invalid_next_codemaker|{next_codemaker},{codemaker}")
                res = f"invalid next codemaker. you must ask {next_codemaker}"
                print(f"W: #{turn + 1} invalid asked codemaker '{codemaker}', next was {next_codemaker}"
                      f" | out_tokens = {trace[-1]['n_tokens_out']}")
            else:
                code = codes_by_name[codemaker]
                res, noisy_guess = noisy_mm.noisy_score(guess, code, args.p_corr)
                trace[-1]["noisy_guess"] = noisy_guess
                print(f"I: #{(turn + 1):3} {codemaker:10} g({guess} -> {noisy_guess}, {code}) = {res}"
                    f" | tokens: in={trace[-1]['n_tokens_in']}, out={trace[-1]['n_tokens_out']}")

        trace[-1]["res"] = res
        if res == "4 0":
            solved.append(next_codemaker)
            trace[-1]["status"] = {"solved" : next_codemaker}

        remaining = [name for name in codemakers if not name in solved]
        if len(remaining) == 0:
            # everything is solved!
            return solved, trace

        # print the list of unsolved once we solve one:
        if next_codemaker in solved:
            print(f"I: {len(remaining)} codemakers remains: {", ".join(remaining)}")

        # pick next opp
        next_codemaker = random.sample(remaining, k=1)[0]

        if len(calls) == 1:
            input_list.append({
                "type": "function_call_output",
                "call_id": calls[0].call_id,
                "output": f"{res} {next_codemaker}",
            })
        else:
            input_list.append({
                "role": "user",
                "content": "Must have exactly ONE tool call." 
            })

    # we exhausted the number of attempts, return what we have
    trace[-1]["status"] = {"error" : "n_turns_max"}

    return solved, trace

def main():
    os.makedirs("logs", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sessions", type=int, default=20, help="How many sessions to play. Each session will contain n_games. Each session is independent, runs sequentially.")
    parser.add_argument("--n_games", type=int, default=4, help="How many codemakers to play against during each session.")
    parser.add_argument("--n_turns_max", type=int, default=50, help="Amortized max number of turns per game. Session with n_games will have n_games * n_turns_max turns total. The turns are shared, so it is allowed to use more on one game and less on other")
    parser.add_argument("--n_tokens_max", type=int, default=2**16, help="Max output tokens per turn, passed to API.")
    parser.add_argument("--p_corr", type=float, default=0.2, help="probability of each digit corruption")
    parser.add_argument("--secret_codes", help="Hardcoded comma-separate list of secret codes to try. Length must == n_games.")
    
    parser.add_argument("--model", "-m", help="Model name to use. If not provided, will query /models endpoint; if there's only one model, will use it.")
    parser.add_argument("--reasoning-effort", default='low')
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--note", help="optional note to store in results. Useful for testing externally configurable options, like llama.cpp server options.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tool-choice", choices=["required", "auto"], default="required",
                        help="'required' forces one make_guess call per turn; use 'auto' for DeepSeek thinking mode")

    args = parser.parse_args()

    if args.base_url is not None:
        print(f"I: base-url override: {args.base_url}")
        if args.api_key is None:
            # To avoid leaking API KEY
            print(f"W: base-url override: won't use OPENAI_API_KEY env var, using 'sk-no-key' as API key. Pass --api-key if needed.")
            api_key = "sk-no-key"
        else:
            api_key = args.api_key
        client = openai.OpenAI(base_url=args.base_url, api_key=api_key, timeout=API_TIMEOUT, max_retries=0)
    else:
        # will try use env var, but still allow to override.
        client = openai.OpenAI(api_key=args.api_key, timeout=API_TIMEOUT, max_retries=0)

    if args.model is None:
        print(f"I: no model specified, checking /models endpoint")
        models = client.models.list().data
        if len(models) == 0:
            print(f"E: no model specified and /models endpoint empty")
            exit(1)
        if len(models) > 1:
            print(f"E: no model specified and /models endpoint has multiple options. Pick one:")
            for model in models:
                print(f"E:    {model.id}")
            exit(1)
        args.model = models[0].id

    print(f"I: model: {args.model}")

    rng = random.Random(args.seed)
    if args.secret_codes is not None:
        codes = args.secret_codes.split(",")
        if len(codes) != args.n_games:
            print(f"E: if passing secret_codes, the length must match n_games. len({codes}) != {args.n_games}")
            exit(1)
        for code in codes:
            if code not in noisy_mm.DATASET:
                print(f"E: {code} is not a valid code")
                exit(1)
        code_set = codes * args.n_sessions
    else:
        # need to generate n_sessions * n_games
        code_set = rng.sample(noisy_mm.DATASET, k=(args.n_sessions * args.n_games))

    #pprint(code_set)

    k = args.n_games

    # each session consists of n_games
    for n in range(args.n_sessions):
        dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        codes = code_set[k * n : k * n + k]
        print(codes)

        # we can probably pass custom rng here.
        solved, trace = run(client=client, secret_codes=codes, args=args)

        content = {
            "solved": solved,
            "model" : args.model,
            "secret_codes": codes,
            "trace" : trace,
            "n_turns" : len(trace),
            "n_tokens_out_total" : sum(l["n_tokens_out"] for l in trace),
            "n_invalid_guesses"  : sum(l.get("res") == "invalid guess" for l in trace),
            "args" : {
                "seed": args.seed,
                "p_corruption": args.p_corr,
                "reasoning_effort": args.reasoning_effort,
                "n_sessions": args.n_sessions,
                "n_games" : args.n_games,
                "tool_choice": args.tool_choice,
                "n_turns_max_total": args.n_turns_max * args.n_games,
                "n_tokens_max" : args.n_tokens_max,
            },
        }
        if args.note is not None:
            content["note"] = args.note
        content_str = json.dumps(content)
        tag = secrets.token_hex(3)
        with open(f"logs/{dt}-{tag}.json", "x") as fw:
            fw.write(content_str)

if __name__ == "__main__":
    main()
