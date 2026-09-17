import datetime
import openai
import json
import random
import argparse

P_CORRUPTION = 0.1
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
Use the provided tool 'make_guess'.
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

If you guessed the correct secret number, you will always get "4 0".

If not, each of the digits in your guess might be corrupted.
Corruption is independent per digit.
p(corruption) for each digit is the same number p = {P_CORRUPTION}.
After corruption, the "Bulls Cows" response will be computed.
Corrupted digit will not match anything, so you might have information loss.

## Example 1:

secret     = 1234
your guess = 1235

after corruption, guess = 1?35

the output you get is "2 0".

## Example 2:

secret     = 1234
your guess = 1234

You guessed correctly, so there's no chance of corruption.
The output you get is "4 0".

## Example 3:

secret = 1234
your guess = 1236

after corruption, guess = 1??6

Two digits were corrupted you'll get "1 0".

## Example 4:

secret     = 1234
your guess = 1238

after corruption, guess = 1238 - no corruption happened in this case.

the output you get is "3 0".

Your turn!
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

# returns result + corrupted version
def make_guess(guess, secret):
    if guess == secret:
        # no corruption possible
        return "4 0", guess

    if guess not in DATASET:
        # guess was not a valid candidate
        return "error", guess

    # do corruption
    guess = "".join('?' if random.random() < P_CORRUPTION else c for c in guess)
    
    bulls = sum(1 if a == b else 0 for a, b in zip(guess, secret))
    cows = len(set(guess).intersection(secret)) - bulls
    result = f"{bulls} {cows}"
    return result, guess

# TODO: for longer context version, we need multiple puzzles per session
def run(secret, model, reasoning_effort, client):
    usage_log = []

    input_list = [{"role": "user", "content": PROMPT}]

    # TODO: we just allow single guess/turn. We need correct accounting here.
    guesses = 0
    guessed = False

    for turn in range(MAX_TURNS):
        response = client.responses.create(input=input_list, model=model, tools=TOOLS, reasoning={"effort" : reasoning_effort})
        usage_log.append(response.usage)
        input_list += response.output

        guesses_by_turn = 0

        for output_item in response.output:
            if output_item.type == "function_call" and output_item.name == "make_guess":
                guesses_by_turn += 1
                guesses += 1
                if guesses_by_turn > 1:
                    print(f"W: more than one guess on turn {turn}.")
                    return False, usage_log, guesses

                args = json.loads(output_item.arguments)
                guess = args["guess"]
                res, corrupted_guess = make_guess(guess, secret)

                if res == "4 0":
                    guessed = True

                if res == "error":
                    print(f"W: invalid guess '{guess}' on turn {turn}.")
                    return False, usage_log, guesses

                print(f"I: turn {turn} make_guess({guess} -> {corrupted_guess}, {secret}) = {res}")

                input_list.append({
                    "type": "function_call_output",
                    "call_id": output_item.call_id,
                    "output": res,
                })

        if guesses_by_turn == 0:
            print(f"W: no guess on turn {turn}.")
            return False, usage_log, guesses

        if guessed:
            return True, usage_log, guesses

    return False, usage_log, guesses

DESC="""
A benchmark/study for LLM models.

Model's task is to play a game of Mastermind, also known as Bulls and Cows.
Our variant of the game has imperfect communication channel.

The goal is to test model itself, not harness. OpenAI responses API is used.

Currently verified to work with OpenAI API and local llama.cpp server.

Model selection logic:
 - if model is specified explicitly, use it;
 - if model is not passed, check models endpoint; if there's exactly one model available, use it
 - if there's 0/more then one model, show error and ask to specify the model
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
        print(f"I: using {args.base_url} base-url override")
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

    print(f"I: using model {model}")

    if args.secret is not None:
        if args.secret not in DATASET:
            print(f"E: provided secret {args.secret} is not a valid secret number")
            exit(1)
        fixed_secret = args.secret

    for n in range(args.samples):
        secret = fixed_secret if fixed_secret is not None else random.sample(DATASET, k=1)[0]
        dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"I: sample {n}/{args.samples} with secret={secret}")
        success, usage_log, guesses = run(secret=secret, model=model, reasoning_effort=args.reasoning_effort, client=client)
        content = {
            "success": success,
            "model" : model,
            "secret": secret,
            "reasoning_effort": args.reasoning_effort,
            "p_corruption": P_CORRUPTION,
            "max_turns": MAX_TURNS,
            "usage_log" : [{
                "completion_tokens": l.output_tokens,
                "prompt_tokens": l.input_tokens
            } for l in usage_log],
            "turns" : len(usage_log),
            "total_gen_tokens" : sum(l.output_tokens for l in usage_log),
            "guesses" : guesses
        }
        content_str = json.dumps(content)
        with open(f"logs/{dt}-{secret}.json", "w") as fw:
            fw.write(content_str)

if __name__ == "__main__":
    main()
