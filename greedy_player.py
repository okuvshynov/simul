import argparse
import datetime
import math
import random
import json
import secrets

import noisy_mm

# Greedy here means 'always going after the potential secret'.
# The sampling of the guess itself can be controlled with temperature.
# This 'greedy' is to contrast with information-gain maximization algorithm.
# Greedy is ok for our purposes of 'reasonable baseline'.

P_CORRUPTION = 0.2

N_TURNS_MAX = 50

# How likely to observe (b, w) if true answer before corruption would be 
# (b_, w_) and p(corruption) = p. Corruption is independent for each digit
def answer_probability(b, w, b_, w_, p):
    # corruption can only reduce b and w.
    if b > b_ or w > w_:
        return 0.0
    pb = math.comb(b_, b) * ((1 - p) ** b) * (p ** (b_ - b))
    pw = math.comb(w_, w) * ((1 - p) ** w) * (p ** (w_ - w))

    return pb * pw

# dataset: all valid secrets
# history: [(guess, b, w)] after corruption
# p: probability of corruption for a digit
def posterior(dataset, history, p_corruption):
    res = {}

    for s in dataset:
        # first, check if this exact secret was asked.
        # if that was the case and we got "4 0" than 
        # it's the only option. The game should have stopped,
        # but the only possible secret is still s.
        # if it was asked, and got anything other than "4 0"
        # it cannot be the secret per rules.
        # 
        # if never asked, update the likelihood

        p = 1.0
        for g, b, w in history:
            if g == s:
                if b == 4 and w == 0:
                    return {s: 1.0}
                else:
                    p = 0.0
            else:
                b_, w_ = noisy_mm.exact_score(g, s)
                p *= answer_probability(b, w, b_, w_, p_corruption)

            if p == 0.0:
                break

        if p > 0.0:
            res[s] = p

    norm = sum(res.values())

    # should never happen
    if norm == 0.0:
        raise ValueError("Inconsistency in the history")

    return {s: v / norm for s, v in res.items() if v / norm > 0.0}

def sample_move(probs, temp=1.0):
    if temp <= 0.0:
        # greedy
        return max(range(len(probs)), key=lambda k: probs[k])

    logits    = [math.log(p) / temp for p in probs]
    max_logit = max(logits)
    weights   = [math.exp(logit - max_logit) for logit in logits]

    return random.choices(range(len(weights)), weights, k=1)[0]
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--secret")
    parser.add_argument("--n_skip", type=int, default=0, help="Skip first seeded secrets. Useful if you want to 'continue from same seed'")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--quiet", "-q", action='store_true')
    parser.add_argument("--note", help="optional note to store in results. Useful for testing externally configurable options, like llama.cpp server options.")
    parser.add_argument("--seed", type=int, default=42, help="seed for secret selection. Corruption/noise is separate.")
    parser.add_argument("--seed-noise", type=int, default=8765, help="seed for noise.")
    args = parser.parse_args()

    turns = []

    if args.secret is not None:
        if args.secret not in noisy_mm.DATASET:
            print(f"E: provided secret {args.secret} is not a valid secret number")
            exit(1)
        secret_set = [args.secret] * args.n_samples
    else:
        rng = random.Random(args.seed)
        secret_set = rng.sample(noisy_mm.DATASET, k=args.n_samples+args.n_skip)[args.n_skip:]

    rng_noise = random.Random(args.seed_noise)

    for i, secret in enumerate(secret_set):
        dt = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        history = []
        turn = 0
        status = "unsolved"

        # now let's play a game
        while True:
            turn += 1
            belief = list(posterior(noisy_mm.DATASET, history, P_CORRUPTION).items())
            probs  = [p for _, p in belief]
            # this player never 'probes' - always choosing one of the 'possible'
            # values, which is suboptimal, but acceptable for this use-case.

            guess, _ = belief[sample_move(probs, args.temp)]
            res, noisy_guess = noisy_mm.noisy_score(guess, secret, P_CORRUPTION, rng_noise=rng_noise)
            if not args.quiet:
                print(f"I: #{turn} g({guess} -> {noisy_guess}, {secret}) = {res}")
            if res == "4 0":
                status = "solved"
                turns.append(turn)
                break
            [b, w] = res.split()
            history.append((guess, int(b), int(w)))

        print(f"I: sample={i + 1} n_turns = {turn}")
        content = {
            "status": status,
            "model" : "greedy",
            "secret": secret,
            "p_corruption": P_CORRUPTION,
            "n_turns_max": N_TURNS_MAX,
            "n_turns" : turn,
            "args" : {
                "seed": args.seed,
                "seed_noise" : args.seed_noise,
                "n_samples": args.n_samples,
                "n_skip": args.n_skip,
                "sample_idx": i, # this means, absolute idx = n + n_skip
                "secret": args.secret,
            },
        }
        if args.note is not None:
            content["note"] = args.note
        content_str = json.dumps(content)
        tag = secrets.token_hex(3)
        with open(f"logs/{dt}-{secret}-{tag}.json", "x") as fw:
            fw.write(content_str)

    print(f"I: global avg n_turns = {sum(turns) / len(turns)}")

if __name__ == "__main__":
    main()