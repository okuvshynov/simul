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

OPPONENTS = [
    ("Red",      "#E6194B"),
    ("Green",    "#3CB44B"),
    ("Yellow",   "#FFE119"),
    ("Blue",     "#4363D8"),
    ("Orange",   "#F58231"),
    ("Purple",   "#911EB4"),
    ("Cyan",     "#42D4F4"),
    ("Magenta",  "#F032E6"),
    ("Lime",     "#BFEF45"),
    ("Pink",     "#FABED4"),
    ("Teal",     "#469990"),
    ("Lavender", "#DCBEFF"),
    ("Brown",    "#9A6324"),
    ("Maroon",   "#800000"),
    ("Olive",    "#808000"),
    ("Mint",     "#AAFFC3"),
]

P_CORRUPTION = 0.2
N_TURNS_MAX  = 200

# default is 10 min. Bump to 2 hours for local runs
API_TIMEOUT  = 7200

# if model keeps thinking for a single turn for 2^16k tokens, that's bad enough.
# TODO: this needs to be configurable
N_TOKENS_PER_TURN_MAX = 2 ** 16

# API calls per turn before giving up on the sample.
N_ATTEMPTS_MAX = 3

PROMPT = f"""
Let's play a game of Noisy Mastermind.

It is a variant of Mastermind game with imperfect communication channel.
You are a grandmaster and play simul session with multiple codemakers.

# Rules

Each codemaker comes up with a secret number.
The number has exactly four distinct digits 0..9.
Digit repetitions are not allowed in this variation - every digit is unique.
First digit cannot be 0, number must start with 1..9.

Examples of valid secret numbers: 1234, 1290, 9081

Examples of invalid secret numbers: 0123, 1111, 9912, 12345, 468

Example procedure to generate all valid secret numbers:

secrets = [str(d) for d in range(1000, 10000) if len(set(str(d))) == 4]

You are the codebreaker, your task is to guess the secret number.

At each turn you make a single guess to one codemaker. 
Your guess must be a potentially valid secret with exact same restrictions - part of the secrets set.
Do not try to cheat by making a guess with repeated characters, non-digits or shorter/longer guesses.

You will be told which codemaker to ask next. You MUST ask the designated next codemaker, otherwise your turn will be wasted.

After every guess, you'll get two numbers and next codemaker to ask:
1. Number of digits you guessed correctly which are in the right position.
This number is often referred to as 'black'.
2. Number of digits you guessed correctly which are in the wrong position;
This number is often referred to as 'white'.
3. Nickname of the next codemaker to ask.

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
p = {P_CORRUPTION}.

The response you get will be computed after corruption.
Corrupted digit will not match anything - it will not be equal to any digit.
Thus, corruption can decrease the number of matches you would get,
but never increase.

# Examples

## Example 1:

secret     = 1234
your guess = 1235

Let's say one digit got corrupted and after corruption guess becomes "1?35"

the output you get is "2 0" - digit "2" was corrupted and ignored.

## Example 2:

secret     = 1234
your guess = 1234

You guessed correctly, so there was no chance of corruption.
The output you get is "4 0".

## Example 3:

secret     = 1234
your guess = 1236

after corruption, guess = 1??6

Two digits were corrupted, you'll get "1 0".

## Example 4:

secret     = 1234
your guess = 1238

after corruption, guess = 1238 - no corruption happened in this case.

the output you get is "3 0".

# Scoring

For every game the number of guesses to solve each problem will be recorded.
You have {N_TURNS_MAX} guesses total; if you guessed only some of the puzzles, you'll get partial score. 

Your goal is:
1. Solve as many puzzles as possible. This is your primary goal.
2. Minimize the average number of guesses for solved puzzles.

If invalid guess is encountered, for example:

- number starting with 0
- repeated digits
- non-digits
- guess with number of digits other than 4
- guess directed to a wrong codemaker

the output will be "invalid guess". The turn is lost and still counts
towards the {N_TURNS_MAX} limit.

If you make no tool calls, or more than one tool call in a turn,
the game stops and you get 0 score.

"""

PROMPT_SUFFIX = """
You play against codemakers: {}.
First codemaker to query is {}.
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

def run(secrets, model, reasoning_effort, tool_choice, client, rng_noise, prompt):
    input_list = [{"role": "user", "content": prompt}]
    trace = []

    # "required" + parallel_tool_calls=False for one call per turn. 
    # DeepSeek rejects it in thinking mode, so "auto" means 'default'.
    extra = {"tool_choice": "required"} if tool_choice == "required" else {}

    k = len(secrets)
    opps = [name for name, _ in OPPONENTS[:k]]
    solved = {name: False for name in opps}
    next_opp = opps[0]
    secrets_by_name = {name: secret for name, secret in zip(opps, secrets)}

    for turn in range(N_TURNS_MAX):
        #pprint(input_list)
        try:
            response, attempts = create_response(
                client,
                input=input_list,
                model=model,
                tools=TOOLS,
                reasoning={"effort" : reasoning_effort},
                max_output_tokens=N_TOKENS_PER_TURN_MAX,
                parallel_tool_calls=False,
                **extra,
            )
        except ApiFailure as e:
            print(f"W: giving up on turn {turn + 1}: {e}")
            trace.append({
                "n_tokens_in"  : 0,
                "n_tokens_out" : 0,
                "n_calls"      : 0,
                "n_attempts"   : e.attempts,
                "status"       : "err_api"
            })
            return trace

        trace.append({
            "n_tokens_in"  : response.usage.input_tokens,
            "n_tokens_out" : response.usage.output_tokens,
            "status"       : "",
            "n_calls"      : 0,
            "n_attempts"   : attempts,
        })
        input_list += response.output

        if response.status != "completed":
            trace[-1]["status"] = "err_response"
            print(f"W: response error, possibly hit {N_TOKENS_PER_TURN_MAX}.")
            trace[-1]["error_log"] = save_error_response(response, turn)
            return trace

        # first, check that we have exactly one guess tool call per instructions
        calls = [
            item for item in response.output
            if item.type == "function_call" and item.name == "make_guess"
        ]

        trace[-1]["n_calls"] = len(calls)

        if len(calls) != 1:
            trace[-1]["status"] = "err_n_calls"
            print(f"W: Expected one guess per turn, got {len(calls)}")
            trace[-1]["error_log"] = save_error_response(response, turn)
            return trace

        try:
            args = json.loads(calls[0].arguments)
            guess = args["guess"]
            codemaker = args["codemaker"]
        except (json.JSONDecodeError, TypeError, KeyError):
            # this way we'll keep invalid guess in the logs for inspection
            guess = calls[0].arguments
            codemaker = "unknown"

        trace[-1]["guess"] = guess
        trace[-1]["codemaker"] = codemaker

        if guess not in noisy_mm.DATASET:
            # the turn is lost, but the game goes on
            res = "invalid guess"
            print(f"W: #{turn + 1} invalid guess '{guess}'"
                  f" | out_tokens = {trace[-1]['n_tokens_out']}")
        elif codemaker != next_opp:
            # the turn is lost, but the game goes on
            res = f"invalid next codemaker. you must ask {next_opp}"
            print(f"W: #{turn + 1} invalid asked codemaker '{codemaker}'"
                  f" | out_tokens = {trace[-1]['n_tokens_out']}")
        else:
            secret = secrets_by_name[codemaker]
            res, noisy_guess = noisy_mm.noisy_score(guess, secret, P_CORRUPTION, rng_noise)
            # rename later after we simplify visualizer
            trace[-1]["corrupted_guess"] = noisy_guess
            print(f"I: #{(turn + 1):3} {codemaker:10} g({guess} -> {noisy_guess}, {secret}) = {res}"
                  f" | out_tokens = {trace[-1]['n_tokens_out']}")

        trace[-1]["res"] = res
        if res == "4 0":
            solved[next_opp] = True

        remaining = [name for name in opps if not solved[name]]
        if len(remaining) == 0:
            # everything is solved!
            trace[-1]["status"] = solved
            return trace

        # pick next opp
        next_opp = random.sample(remaining, k=1)[0]

        input_list.append({
            "type": "function_call_output",
            "call_id": calls[0].call_id,
            "output": f"{res} {next_opp}",
        })

    # we exhausted the number of attempts, return what we have
    trace[-1]["status"] = solved

    return trace

def main():
    os.makedirs("logs_gm", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--model", "-m")
    parser.add_argument("--reasoning-effort", default='low')
    parser.add_argument("--secrets")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--note", help="optional note to store in results. Useful for testing externally configurable options, like llama.cpp server options.")
    parser.add_argument("--seed", type=int, default=42, help="seed for secret selection. Corruption/noise is separate.")
    parser.add_argument("--seed-noise", type=int, default=8765, help="seed for noise.")

    parser.add_argument("--n_skip", type=int, default=0, help="Skip first seeded secrets. Useful if you want to 'continue from same seed'")

    parser.add_argument("--tool-choice", choices=["required", "auto"], default="required",
                        help="'required' forces one make_guess call per turn; use 'auto' for DeepSeek thinking mode")
    parser.add_argument("--n_simul", type=int, default=4, help="How many codemakers to play against")

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
        model = models[0].id
    else:
        model = args.model

    print(f"I: model: {model}")

    if args.secrets is not None:
        codes = args.secrets.split(",")
        if len(codes) != args.n_simul:
            print(f"E: if passing secrets, the length must match n_simul. len({secrets}) != {args.n_simul}")
            exit(1)
        for secret in codes:
            if secret not in noisy_mm.DATASET:
                print(f"E: provided secret {secret} is not a valid secret number")
                exit(1)
        secret_set = codes * args.n_samples
    else:
        rng = random.Random(args.seed)
        # need to generate n_samples * n_simul and skip n_skip * n_simul
        secret_set = rng.sample(noisy_mm.DATASET, k=(args.n_samples+args.n_skip) * args.n_simul)[args.n_skip * args.n_simul:]

    rng_noise = random.Random(args.seed_noise)

    #pprint(secret_set)

    k = args.n_simul
    codemakers = [name for name, _ in OPPONENTS[:k]]
    prompt = PROMPT + PROMPT_SUFFIX.format(", ".join(codemakers), codemakers[0])

    print(prompt)
    # each sample is a game of n_simul puzzles
    for n in range(args.n_samples):
        dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        codes = secret_set[k * n : k * n + k]
        print(codes)
        trace = run(secrets=codes, model=model, reasoning_effort=args.reasoning_effort,
                    tool_choice=args.tool_choice, client=client, rng_noise=rng_noise, prompt=prompt)

        content = {
            "status": trace[-1]["status"],
            "model" : model,
            "secrets": codes,
            "reasoning_effort": args.reasoning_effort,
            "p_corruption": P_CORRUPTION,
            "n_turns_max": N_TURNS_MAX,
            "trace" : trace,
            "n_turns" : len(trace),
            "n_tokens_out_total" : sum(l["n_tokens_out"] for l in trace),
            "n_invalid_guesses"  : sum(l.get("res") == "invalid guess" for l in trace),
            "args" : {
                "seed": args.seed,
                "seed_noise" : args.seed_noise,
                "n_samples": args.n_samples,
                "n_skip": args.n_skip,
                "sample_idx": n, # this means, absolute idx = n + n_skip
                "tool_choice": args.tool_choice
            },
        }
        if args.note is not None:
            content["note"] = args.note
        content_str = json.dumps(content)
        tag = secrets.token_hex(3)
        with open(f"logs_gm/{dt}-{tag}.json", "x") as fw:
            fw.write(content_str)

if __name__ == "__main__":
    main()
