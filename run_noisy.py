import datetime
import openai
import json
import random
import argparse
import sys

client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="sk-no-key")

P_CORRUPTION = 0.1
DATASET = [str(d) for d in range(1000, 10000) if len(set(str(d))) == 4]

PROMPT = f"""
Let's play a game of Bulls and Cows/Mastermind with imperfect communication channel.

# Rules

Your task is to guess the secret number. The secret number has four distinct digits 0..9 with no repetitions, cannot start with 0.
Use the provided tool 'make_guess'.
The output you get is a string "Bulls Cows".
"4 0" would indicate solved problem.

# Scoring

For every game instance the number of guesses to solve the problem will be recorded, and average across games will be taken.
Your goal is to minimize that average.

# Restrictions

Your guess must be a potentially valid solution to the puzzle. Do not try to game the system by making a guess with repeated/non-digit characters. You'll get 'error' as tool result and it will be still counted towards your turn count. 
Do single guess per turn. Making multiple tool calls will make evaluation harder; if you pass multiple tool calls, you'll only get result for first one. 

# Communication.

If you guessed the correct secret number, you will always get "4 0".

If not, before evaluating your guess, each of the digits in your guess might be corrupted. 
Corruption is independent per digits, p(corruption) for each digit is the same number p = {P_CORRUPTION}.
Corrupted digit is not recoverable and will be ignored when computing bulls/cows.

## Example 1:

secret = 1234
your guess = 1235

after corruption, guess = 1?35

the output you get is "2 0".

## Example 2:

secret = 1234
your guess = 1234

You guessed correctly, so there's no chance of corruption. the output you get is "4 0".

## Example 3:

secret = 1234
your guess = 1236

after corruption:

guess = 1??6

Two digits were corrupted you'll get "1 0".

Your turn.
"""

TOOLS = [
    {
        "type" : "function",
        "function" : {
            "name" : "make_guess",
            "description" : "Make a guess in a game. The input is four digit guess, the output is a single string with two numbers - bulls & cows respectively, taking into account probabilistic corruption.",
            "parameters" : {
                "type" : "object",
                "properties" : {
                    "guess": {
                        "type" : "string",
                        "description" : "four digit guess for the game. Your guess must be a valid potential solution to the puzzle - 4 digits, no repeated digits, not starting with 0, no other symbols. If it is invalid, you'll get 'error' string as a result."
                    }
                },
                "required" : ["guess"]
            }
        }
    }
]

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

def run(secret, max_turns, model, reasoning_effort):
    log = []

    messages = [{"role": "user", "content": PROMPT}]

    for turn in range(max_turns):
        # TODO: test setting reasoning_effort from client
        response = client.chat.completions.create(messages=messages, model=model, tools=TOOLS, reasoning_effort=reasoning_effort)

        log.append(response.usage)

        message = response.choices[0].message

        if len(message.tool_calls) != 1:
            print(f"Got {len(message.tool_calls)} tool calls in a single message.")

        tool_call = message.tool_calls[0]

        if tool_call.function.name == "make_guess":
            args = json.loads(tool_call.function.arguments)
            guess = args["guess"]
            res, corrupted_guess = make_guess(guess, secret)

            print(f"turn {turn} make_guess({guess} -> {corrupted_guess}, {secret}) = {res}")

            assistant_message = {
                "role": "assistant",
                "content": message.content,
                "tool_calls" : [tool_call],
            }

            reasoning_content = getattr(message, "reasoning_content", None)

            if reasoning_content:
                assistant_message["reasoning_content"] = reasoning_content

            tool_reply = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content" : res
            }

            messages.append(assistant_message)
            messages.append(tool_reply)
            if res == "4 0":
                return True, log
        else:
            print("! no make_guess tool call.")
            return False, log

    return False, log

def check_model():
    models = client.models.list().data
    if len(models) != 1:
        print(f"! expected server to have single model, got {models}")
        sys.exit(1)
    return models[0].id

def main():
    parser = argparse.ArgumentParser("Solving Bulls & Cows")
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--model", "-m")
    parser.add_argument("--reasoning-effort", default='max')
    args = parser.parse_args()

    model = args.model if args.model is not None else check_model()
    print(f"? model = {model}")

    for n in range(args.samples):
        secret = random.sample(DATASET, k=1)[0]
        dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"sample {n}/{args.samples} with secret={secret}")
        success, log = run(secret=secret, max_turns=args.max_turns, model=model, reasoning_effort=args.reasoning_effort)
        content = {
            "success": success,
            "model" : model,
            "secret": secret,
            "reasoning_effort": args.reasoning_effort,
            "p_corruption": P_CORRUPTION,
            "usage" : [{
                "completion_tokens": l.completion_tokens,
                "prompt_tokens": l.prompt_tokens
            } for l in log] 
        }
        content_str = json.dumps(content)
        with open(f"logs/{dt}-{secret}", "w") as fw:
            fw.write(content_str)

if __name__ == "__main__":
    main()
