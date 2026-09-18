import datetime
import openai
import json
import random
import argparse

P_CORRUPTION = 0.2
MAX_TURNS    = 50
DATASET      = [str(d) for d in range(1000, 10000) if len(set(str(d))) == 4]

PROMPT = f"""
Let's play a game of Mastermind, also known as Bulls and Cows.
Our variant of the game has imperfect communication channel.

# Rules

Your task is to guess the secret number.
The number has exactly four distinct digits 0..9.
Digit repetitions are not allowed.
First digit cannot be 0.
Use the provided tool 'make_guess' to make a guess.
The output you get is a string "Bulls Cows".
Bulls: how many digits are correct and are on the right spot.
Cows: how many digits are correct but are on the wrong spot.

"4 0" would indicate solved problem.

# Scoring

For every game the number of guesses to solve the problem will be recorded.
If you fail to find a solution in {MAX_TURNS} guesses, the puzzle is marked as 
unsolved. Your priorities are (in order):

1. Solve as many puzzles as possible.
2. Minimize the average number of guesses for solved ones.

# Restrictions

Your guess must be a potentially valid solution to the puzzle.
Do not try to cheat by making a guess with repeated/non-digit characters.
You MUST make EXACTLY ONE tool call per turn.

If invalid guess is encountered for any of the reasons:

- number starting with 0
- repeated digits
- non-digits
- no tool calls
- more then one tool call

the entire puzzle will be counted as unsolved.

# Communication corruption.

If you guessed the secret number, you will always get "4 0".

If not, each of the digits in your guess might be corrupted before comparison.
Corruption is independent for each digit.
p(corruption) for each digit is the same number p = {P_CORRUPTION}.
After corruption, the "Bulls Cows" response will be computed.
Corrupted digit will not match anything, so you might have information loss.

## Example 1:

secret     = 1234
your guess = 1235

Let's say one digit got corrupted and after corruption guess = 1?35

the output you get is "2 0" - digit "2" was corrupted and not counted.

## Example 2:

secret     = 1234
your guess = 1234

You guessed correctly, so there's no chance of corruption.
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

Your turn, make a first guess.
"""

TOOLS = [{
    "type" : "function",
    "name" : "make_guess",
    "description" : "Make a guess. The output is a string with two numbers: 'Bulls Cows'",
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
def make_guess(guess, secret):
    if guess == secret:
        # no corruption if guessed correctly;
        return "4 0", guess

    if guess not in DATASET:
        # This should never happen as check is external. 
        print("E: bug: guess was not in DATASET")
        exit(1)

    # do corruption
    guess = "".join('?' if random.random() < P_CORRUPTION else c for c in guess)
    
    bulls  = sum(1 if a == b else 0 for a, b in zip(guess, secret))
    cows   = len(set(guess).intersection(secret)) - bulls
    result = f"{bulls} {cows}"

    return result, guess

def run(secret, model, reasoning_effort, client):
    usage_log = []

    input_list = [{"role": "user", "content": PROMPT}]

    guessed = False
    trace = []

    for turn in range(MAX_TURNS):

        response = client.responses.create(
            input=input_list,
            model=model,
            tools=TOOLS,
            reasoning={"effort" : reasoning_effort}
        )

        usage_log.append(response.usage)
        trace.append({
            "input_tokens"  : response.usage.input_tokens,
            "output_tokens" : response.usage.output_tokens,
            "status" : ""   
        })
        input_list += response.output

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

        args = json.loads(calls[0].arguments)
        guess = args["guess"]

        trace[-1]["guess"] = guess

        if guess not in DATASET:
            trace[-1]["status"] = "err_inv_guess"
            print(f"W: invalid guess '{guess}' on turn {turn}.")
            return trace

        res, corrupted_guess = make_guess(guess, secret)
        print(f"I: #{turn} g({guess} -> {corrupted_guess}, {secret}) = {res}")

        trace[-1]["corrupted_guess"] = corrupted_guess
        trace[-1]["res"] = res
        if res == "4 0":
            trace[-1]["status"] = "solved"
            return trace

        input_list.append({
            "type": "function_call_output",
            "call_id": calls[0].call_id,
            "output": res,
        })



    return trace

DESC="""
Model's task is to play a game of Mastermind, also known as Bulls and Cows.
Our variant of the game has imperfect communication channel.

The goal is to test model itself, not harness. OpenAI responses API is used.

Currently verified to work with OpenAI API and local llama.cpp server.
"""

def main():
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--model", "-m")
    parser.add_argument("--reasoning-effort", default='high')
    parser.add_argument("--secret")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")

    args = parser.parse_args()

    if args.base_url is not None:
        print(f"I: base-url override: {args.base_url}")
        if args.api_key is None:
            # To avoid leaking API KEY
            print(f"W: base-url override: won't use OPENAI_API_KEY env var, using 'sk-no-key' as API key. Pass --api-key if needed.")
            api_key = "sk-no-key"
        else:
            api_key = args.api_key
        client = openai.OpenAI(base_url=args.base_url, api_key=api_key)
    else:
        # will try use env var, but still allow to override.
        client = openai.OpenAI(api_key=args.api_key)

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

    fixed_secret = None
    if args.secret is not None:
        if args.secret not in DATASET:
            print(f"E: provided secret {args.secret} is not a valid secret number")
            exit(1)
        fixed_secret = args.secret

    for n in range(args.samples):
        secret = fixed_secret if fixed_secret is not None else random.sample(DATASET, k=1)[0]
        dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"I: sample {n}/{args.samples} with secret={secret}")
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
            "total_gen_tokens" : sum(l["output_tokens"] for l in trace)
        }
        content_str = json.dumps(content)
        with open(f"logs/{dt}-{secret}.json", "w") as fw:
            fw.write(content_str)

if __name__ == "__main__":
    main()
