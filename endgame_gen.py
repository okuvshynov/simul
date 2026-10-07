# there must be two parts: history generation and LLM benchmark
# generation can be done independently, so, we, say, generate 1000 independent games;

import argparse

import noisy_mm
import random
import json
import secrets
from greedy_player import posterior, sample_move

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, help="seed for secret selection.")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--quiet", "-q", action='store_true')
    parser.add_argument("--p_corr", type=float, default=0.2, help="probability of each digit corruption")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    secret_set = rng.sample(noisy_mm.DATASET, k=args.n_samples)

    for i, secret in enumerate(secret_set):
        history = []
        turn = 0
        sample = {
            "secret" : secret,
            "temp" : args.temp,
            "p_corr" : args.p_corr,
            "turns" : []
        }
        while True:
            turn += 1
            belief = list(posterior(noisy_mm.DATASET, history, args.p_corr).items())
            probs  = [p for _, p in belief]

            guess, _ = belief[sample_move(probs, args.temp)]
            res, noisy_guess = noisy_mm.noisy_score(guess, secret, args.p_corr)
            
            sample["turns"].append({
                "guess" : guess,
                "reply" : res,
                "noisy_guess" : noisy_guess,
                "valid_set_size" : len(probs),
            })
            
            if not args.quiet:
                print(f"I: #{turn} g({guess} -> {noisy_guess}, {secret}) = {res}, |probs| = {len(probs)}")
            if not args.quiet and len(probs) < 5:
                print(" ".join([f"{b}:{p:.3}" for b, p in belief]))
            
            if res == "4 0":
                break
            [b, w] = res.split()
            history.append((guess, int(b), int(w)))

        content_str = json.dumps(sample)
        tag = secrets.token_hex(3)
        with open(f"samples/{secret}-{tag}.json", "x") as fw:
            fw.write(content_str)

if __name__ == '__main__':
    main()