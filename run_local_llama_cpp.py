import datetime
import openai
import json
import random
import argparse
import sys

client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="sk-no-key")

PROMPT = """
Let's play a game of Bulls and Cows.
Your task is to guess the secret number. Use the provided tool 'make_guess'.
The secret number is four distinct digits, cannot start with 0. The output you get is a string "Bulls Cows".
"4 0" would indicate solved problem.

Your turn.
"""

TOOLS = [
    {
        "type" : "function",
        "function" : {
            "name" : "make_guess",
            "description" : "Make a guess in a game. The input is four digit guess, the output is a single string with two numbers - bulls & cows respectively. If your guess is wrong you'll get a single 'error' string.",
            "parameters" : {
                "type" : "object",
                "properties" : {
                    "guess": {
                        "type" : "string",
                        "description" : "four digit guess for the game. Must have 4 digits and not start with 0."
                    }
                },
                "required" : ["guess"]
            }
        }
    }
]

def make_guess(guess, secret):
    if len(guess) != 4:
        return "error"
    if guess[0] == "0":
        return "error"
    bulls = 0
    cows = 0
    for a, b in zip(guess, secret):
        if a == b:
            bulls += 1

    guess_set = set(guess)

    cows = len(guess_set.intersection(secret)) - bulls

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
    args = parser.parse_args()

    model = args.model if args.model is not None else check_model()
    print(model)

    sys.exit(1) 

    dataset = [str(d) for d in range(1234, 10000) if len(set(str(d))) == 4]

    for n in range(args.samples):
        secret = random.sample(dataset, k=1)[0]
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
