import datetime
import openai
import json
import random
import argparse
import sys

client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="sk-no-key")

P_CORRUPTION = 0.1
DATASET = [str(d) for d in range(1234, 10000) if len(set(str(d))) == 4]
SEED = 4242

PROMPT = f"""
Let's play a game of Bulls and Cows/Mastermind with imperfect communication channel.
Your task is to guess the secret number. Use the provided tool 'make_guess'.
The secret number is four distinct digits, cannot start with 0.
The output you get is a string "Bulls Cows".
"4 0" would indicate solved problem.

Communication.

If you guessed the correct secret number, you will always get "4 0".

If not, before evaluation, each of the digits might be corrupted. 
Corruption is independent, p(corruption) for each digit is the same number p = {P_CORRUPTION}.
Corrupted digit is not recoverable and will be ignored when computing bulls/cows.

### Example 1:

secret = 1234
your guess = 1235

after corruption, guess = 1?35

the output you get is "2 0"

### Example 2:

secret = 1234
your guess = 1234

You guessed correctly, so there's no chance of corruption.

### Example 3:

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

def make_guess(guess, secret):
    print(f"guess = {guess}")
    if guess == secret:
        # no corruption possible
        return "4 0"

    if guess not in DATASET:
        # guess was not a valid candidate
        return "error"

    # do corruption
    guess = "".join('?' if random.random() < P_CORRUPTION else c for c in guess)

    print(f"cguess = {guess}")
    bulls = sum(1 if a == b else 0 for a, b in zip(guess, secret))
    cows = len(set(guess).intersection(secret)) - bulls
    return "{} {}".format(bulls, cows)

def run(secret, max_turns, model):
    log = []

    messages = [{"role": "user", "content": PROMPT}]

    for turn in range(max_turns):
        print(f"  turn {turn}")
        # TODO: reasoning_effort
        response = client.chat.completions.create(messages=messages, model=model, tools=TOOLS)

        log.append(response.usage)

        message = response.choices[0].message
        tool_call = message.tool_calls[0]

        if tool_call.function.name == "make_guess":
            args = json.loads(tool_call.function.arguments)
            res = make_guess(args["guess"], secret)

            if res == "4 0":
                print("Success!")
                return True, log

            print("make_guess({}) = {}".format(args["guess"], res))

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
        else:
            print("No make_guess tool call.")
            return False, log

    return False, log

def check_model():
    models = client.models.list().data
    if len(models) != 1:
        print(f"Expected server to have single model, got {models}")
        sys.exit(1)
    return models[0].id

def main():
    parser = argparse.ArgumentParser("Solving Bulls & Cows")
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--model", "-m")
    parser.add_argument("--tag", default="")
    parser.add_argument("--seed", default=SEED, type=int)   
    args = parser.parse_args()

    model = args.model if args.model is not None else check_model()
    print(model)
    random.seed(args.seed)

    for n in range(args.samples):
        secret = random.sample(DATASET, k=1)[0]
        dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"{dt} starting sample {n} with secret={secret}")
        success, log = run(secret=secret, max_turns=args.max_turns, model=model)
        content = {
            "success": success,
            "model" : model,
            "usage" : [{
                "completion_tokens": l.completion_tokens,
                "prompt_tokens": l.prompt_tokens
            } for l in log] 
        }
        content_str = json.dumps(content)
        with open(f"logs/{dt}-{model}-{args.tag}-{secret}", "w") as fw:
            fw.write(content_str)

if __name__ == "__main__":
    main()
