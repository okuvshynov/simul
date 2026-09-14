import datetime
import openai
import json
import random

client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="sk-no-key")

MAX_TURNS = 50
SAMPLES = 20

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

def run(secret):

    log = []

    messages = [{"role": "user", "content": PROMPT}]

    for turn in range(MAX_TURNS):
        print(f"  turn {turn}")
        # TODO: reasoning_effort
        response = client.chat.completions.create(messages=messages, model="GLM-5.3-Flash-UD-IQ3_XXS", tools=TOOLS)

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

def main():
    #secret = "4195"

    dataset = [str(d) for d in range(1234, 10000) if len(set(str(d))) == 4]

    for n in range(SAMPLES):
        secret = random.sample(dataset, k=1)[0]
        dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"{dt} starting sample {n} with secret={secret}")
        success, log = run(secret=secret)
        content = {
            "success": success,
            "usage" : [{
                "completion_tokens": l.completion_tokens,
                "prompt_tokens": l.prompt_tokens
            } for l in log] 
        }
        content_str = json.dumps(content)
        with open(f"logs/{dt}.{secret}", "w") as fw:
            fw.write(content_str)

if __name__ == "__main__":
    main()