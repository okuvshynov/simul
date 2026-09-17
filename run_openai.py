import datetime
import openai
import json
import random
import argparse

from moo import DATASET, PROMPT, MAX_TURNS, TOOLS_OPENAI, make_guess, P_CORRUPTION

client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="sk-no-key")

def run(secret, model, reasoning_effort):
    usage_log = []

    input_list = [{"role": "user", "content": PROMPT}]

    # there could be multiple tool calls per turn, which is suboptimal, but we need to count it
    guesses = 0
    guessed = False

    for turn in range(MAX_TURNS):
        response = client.responses.create(input=input_list, model=model, tools=TOOLS_OPENAI, reasoning={"effort" : reasoning_effort})
        usage_log.append(response.usage)
        input_list += response.output

        for output_item in response.output:
            if output_item.type == "function_call" and output_item.name == "make_guess":
                args = json.loads(output_item.arguments)
                guess = args["guess"]
                res, corrupted_guess = make_guess(guess, secret)
                if res == "4 0":
                    guessed = True
                guesses += 1

                print(f"turn {turn} make_guess({guess} -> {corrupted_guess}, {secret}) = {res}")

                input_list.append({
                    "type": "function_call_output",
                    "call_id": output_item.call_id,
                    "output": res,
                })

        if guessed:
            return True, usage_log, guesses

    return False, usage_log, guesses

def main():
    parser = argparse.ArgumentParser("Solving Bulls & Cows")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--model", "-m", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default='high')
    parser.add_argument("--secret")
    args = parser.parse_args()

    model = args.model
    print(f"model: {model}")

    if args.secret is not None:
        if args.secret not in DATASET:
            print(f"provided secret {args.secret} is not a valid secret number")
            exit(1)
        fixed_secret = args.secret

    for n in range(args.samples):
        secret = fixed_secret if fixed_secret is not None else random.sample(DATASET, k=1)[0]
        dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"sample {n}/{args.samples} with secret={secret}")
        success, usage_log, guesses = run(secret=secret, model=model, reasoning_effort=args.reasoning_effort)
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
