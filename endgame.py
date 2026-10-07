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
from collections import Counter
from pathlib import Path


# default is 10 min. Bump to 2 hours for local runs
API_TIMEOUT  = 7200

# API calls per turn before giving up on the sample.
N_ATTEMPTS_MAX = 3

COLORS = [
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

def format_prompt(p_corr, codemakers, history):
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

At each turn you make a single guess to one codemaker. 
Your guess must be a potentially valid code with exact same restrictions.
Do not try to cheat by making a guess with repeated characters, 
non-digits or shorter/longer guesses. Such turns will be wasted.

You play against multiple codemakers simultaneously.
After every guess, you'll get two numbers:
1. Number of digits you guessed correctly which are in the right position.
This number is often referred to as 'black'.
2. Number of digits you guessed correctly which are in the wrong position;
This number is often referred to as 'white'.

Examples of "black white":
"0 0" - you guessed nothing
"0 4" - you guessed every digit, but it's completely wrong permutation
"4 0" - you solved the puzzle

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

You are provided with a history of previous queries.
History will be formatted as
CodemakerName Guess Black White

Your job is to make a final guess to each codemaker in the same function call.
Use provided make_guesses function.
This is a final opportunity to solve the puzzles and the 
score will be the number of secrets correctly guessed.
There's no value in probing anything - you won't 
have another chance to make a guess. There's no penalty for wrong guess,
but you must submit a single guess for each codemaker.
If more than one guess is submitted for a codemaker, all of them will be ignored.

Codemakers list: {codemakers}.

Full available history:

{history}
"""

def shuffle_merge(lists, rng=random):
    labels = [i for i, l in enumerate(lists) for _ in l]
    rng.shuffle(labels)
    iters = [iter(l) for l in lists]
    return [next(iters[i]) for i in labels]

TOOL_DESCRIPTION = """
You need to provide final guesses for each Codemaker.
Provide a single four-digit guess for each Codemaker.
"""

TOOLS = [{
    "type" : "function",
    "name" : "make_guesses",
    "description" : TOOL_DESCRIPTION,
    "parameters" : {
        "type" : "object",
        "properties" : {
            "guesses": {
                "type" : "array",
                "items" : {
                    "type": "object",
                    "properties": {
                        "codemaker": {"type": "string"},
                        "guess": {"type": "string"}
                    },
                    "required": ["codemaker", "guess"],
                    "additionalProperties": False
                },
            },
        },
        "required" : ["guesses"]
    }
}]

class ApiFailure(Exception):
    """Gave up on one turn: retries exhausted, or a client error that retrying can't fix."""
    def __init__(self, err, attempts):
        super().__init__(str(err))
        self.attempts = attempts


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

def run(client, samples, args):
    #pprint(samples)
    n_games = len(samples)
    codemakers = [c for c, _ in COLORS[:n_games]]
    games = {c : s for c, s in zip(codemakers, samples)}
    answers = {c : s['secret'] for c, s in zip(codemakers, samples)}

    records = []
    for c, s in games.items():
        records.append([f"{c} {t['guess']} {t['reply']}" for t in s['turns'][:-1]])
    history = shuffle_merge(records)

    # TODO: we might have different rate for different samples
    prompt = format_prompt(0.2, ",".join(codemakers), "\n".join(history))
    #pprint(prompt)
    input_list = [{"role": "user", "content": prompt}]
    extra = {"tool_choice": "required"} if args.tool_choice == "required" else {}

    res = {
        "n_solved"     : 0,
    }
    
    
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
        print(f"W: giving up: {e}")
        return res

    if response.status != "completed":
        print(f"W: response error, possibly hit {args.n_tokens_max}.")
        return res

    res["n_tokens_in"] = response.usage.input_tokens
    res["n_tokens_out"] = response.usage.output_tokens

    # check that we have exactly one guess tool call per instructions
    calls = [
        item for item in response.output
        if item.type == "function_call" and item.name == "make_guesses"
    ]

    if len(calls) != 1:
        print(f"W: Expected one guess tool call per turn, got {len(calls)}")
        return res

    try:
        call_args = json.loads(calls[0].arguments)
        guesses = call_args["guesses"]
        pprint(guesses)
        pprint(answers)
        
        for c, a in answers.items():
            gg = [g["guess"] for g in guesses if g["codemaker"] == c]
            if len(gg) != 1:
                pprint(f"W: {c}: {gg}")
                continue
            if gg[0] == a:
                res['n_solved'] += 1


    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"E: {e}")
        return res

    return res

def get_n_samples(n, rng=random.random):
    files = sorted(Path("samples").glob("*.json"))
    return rng.sample(files, n)

def main():
    os.makedirs("eg_logs", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1)    
    parser.add_argument("--n_games", type=int, default=4)
    parser.add_argument("--model", "-m", help="Model name to use. If not provided, will query /models endpoint; if there's only one model, will use it.")
    parser.add_argument("--reasoning-effort", default='low')
    parser.add_argument("--base-url")
    parser.add_argument("--n_tokens_max", type=int, default=2**16, help="Max output tokens per turn, passed to API.")
    parser.add_argument("--api-key")
    parser.add_argument("--seed")
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
    for i in range(args.n_samples):
        print(f"Sample {i + 1}/{args.n_samples}")
        paths = get_n_samples(args.n_games, rng)
        samples = []
        for p in paths:
            with open(p, 'r') as file:
                samples.append(json.load(file))

        res = run(client, samples, args)

        content = {
            "args" : {
                "model": args.model,
                "n_samples" : args.n_samples,
                "n_games": args.n_games,
                "reasoning_effort": args.reasoning_effort,
            },
            "result" : res,
        }
        content_str = json.dumps(content)
        tag = secrets.token_hex(3)
        dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        with open(f"eg_logs/{dt}-{tag}.json", "x") as fw:
            fw.write(content_str)

        print(f"Result: {res}")


if __name__ == "__main__":
    main()