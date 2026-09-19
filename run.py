import argparse
import datetime
import json
import openai
import os
import random
import secrets

P_CORRUPTION = 0.2
MAX_TURNS    = 50
DATASET      = [str(d) for d in range(1000, 10000) if len(set(str(d))) == 4]

# default is 10 min. Bump to 2 hours.
API_TIMEOUT  = 7200

# if model keeps thinking for a single turn for 64k tokens, that's bad enough
MAX_TOKENS   = 2 ** 16

PROMPT = f"""
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

For every game the number of guesses to solve the problem will be recorded.
If you fail to find a solution in {MAX_TURNS} guesses, the puzzle is marked as 
unsolved. Your priorities are (in order):

1. Solve as many puzzles as possible.
2. Minimize the average number of guesses for solved ones.

If invalid guess is encountered, for example:

- number starting with 0
- repeated digits
- non-digits
- guess with number of digits other than 4

the output will be "invalid guess". The turn is lost and still counts
towards the {MAX_TURNS} limit.

If you make no tool calls, or more than one tool call in a turn,
the entire puzzle will be counted as unsolved.


Your turn, make a first guess.
"""

TOOLS = [{
    "type" : "function",
    "name" : "make_guess",
    "description" : "Make a guess. The output is a string with two numbers: 'black white'",
    "parameters" : {
        "type" : "object",
        "properties" : {
            "guess": {
                "type" : "string",
                "description" : "Four digit guess for the game."
            }
        },
        "required" : ["guess"]
    }
}]

# returns result + corrupted version.
def score_guess(guess, secret):
    if guess == secret:
        # no corruption if guessed correctly;
        return "4 0", guess

    # do corruption
    guess = "".join('?' if random.random() < P_CORRUPTION else c for c in guess)
    
    black  = sum(a == b for a, b in zip(guess, secret))
    white  = len(set(guess).intersection(secret)) - black
    result = f"{black} {white}"

    return result, guess

def run(secret, model, reasoning_effort, client):
    input_list = [{"role": "user", "content": PROMPT}]
    trace = []

    for turn in range(MAX_TURNS):
        try:
            response = client.responses.create(
                input=input_list,
                model=model,
                tools=TOOLS,
                reasoning={"effort" : reasoning_effort},
                max_output_tokens=MAX_TOKENS,
                # some models (gpt-5.6-sol) otherwise reply with text after the
                # first tool result and stop; force exactly one call per turn.
                tool_choice="required",
                parallel_tool_calls=False,
            )
        except openai.APIError as e:
            print(f"W: API error on turn {turn + 1}: {e}")
            trace.append({
                "input_tokens" : 0,
                "output_tokens": 0,
                "n_calls"      : 0,
                "status"       : "err_api"
            })
            return trace

        trace.append({
            "input_tokens"  : response.usage.input_tokens,
            "output_tokens" : response.usage.output_tokens,
            "status"        : "",
            "n_calls"       : 0,   
        })
        input_list += response.output

        if response.status != "completed":
            trace[-1]["status"] = "err_response"
            print(f"W: response error, possibly hit {MAX_TOKENS}.")
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
            return trace

        try:
            guess = json.loads(calls[0].arguments)["guess"]
        except (json.JSONDecodeError, TypeError, KeyError):
            guess = calls[0].arguments

        trace[-1]["guess"] = guess

        if guess not in DATASET:
            # the turn is lost, but the game goes on
            res = "invalid guess"
            print(f"W: #{turn + 1} invalid guess '{guess}'"
                  f" | out_tokens = {trace[-1]['output_tokens']}")
        else:
            res, corrupted_guess = score_guess(guess, secret)
            trace[-1]["corrupted_guess"] = corrupted_guess
            print(f"I: #{turn + 1} g({guess} -> {corrupted_guess}, {secret}) = {res}"
                  f" | out_tokens = {trace[-1]['output_tokens']}")

        trace[-1]["res"] = res
        if res == "4 0":
            trace[-1]["status"] = "solved"
            return trace

        input_list.append({
            "type": "function_call_output",
            "call_id": calls[0].call_id,
            "output": res,
        })

    # we exhausted the number of attempts
    trace[-1]["status"] = "unsolved"

    return trace

def main():
    os.makedirs("logs", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--model", "-m")
    parser.add_argument("--reasoning-effort", default='high')
    parser.add_argument("--secret")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.base_url is not None:
        print(f"I: base-url override: {args.base_url}")
        if args.api_key is None:
            # To avoid leaking API KEY
            print(f"W: base-url override: won't use OPENAI_API_KEY env var, using 'sk-no-key' as API key. Pass --api-key if needed.")
            api_key = "sk-no-key"
        else:
            api_key = args.api_key
        client = openai.OpenAI(base_url=args.base_url, api_key=api_key, timeout=API_TIMEOUT)
    else:
        # will try use env var, but still allow to override.
        client = openai.OpenAI(api_key=args.api_key, timeout=API_TIMEOUT)

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

    if args.secret is not None:
        if args.secret not in DATASET:
            print(f"E: provided secret {args.secret} is not a valid secret number")
            exit(1)
        secret_set = [args.secret] * args.samples
    else:
        secret_set = random.Random(args.seed).sample(DATASET, k=args.samples)

    for n, secret in enumerate(secret_set):
        dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        print(f"I: sample {n + 1}/{args.samples} with secret={secret}")
        trace = run(secret=secret, model=model, reasoning_effort=args.reasoning_effort, client=client)
        content = {
            "status": trace[-1]["status"],
            "model" : model,
            "secret": secret,
            "reasoning_effort": args.reasoning_effort,
            "p_corruption": P_CORRUPTION,
            "max_turns": MAX_TURNS,
            "trace" : trace,
            "turns" : len(trace),
            "total_gen_tokens" : sum(l["output_tokens"] for l in trace),
            "invalid_guesses"  : sum(l.get("res") == "invalid guess" for l in trace),
            "args" : {"seed": args.seed, "samples": args.samples, "sample": n, "secret": args.secret},
        }
        content_str = json.dumps(content)
        tag = secrets.token_hex(3)
        with open(f"logs/{dt}-{secret}-{tag}.json", "x") as fw:
            fw.write(content_str)

if __name__ == "__main__":
    main()
