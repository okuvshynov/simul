import math
import random
import sys
import argparse

import noisy_mm

# Greedy here means 'always going after the potential secret'.
# The sampling of the guess itself can be controlled with temperature.
# This 'greedy' is to contrast with information-gain maximization algorithm.
# Greedy is ok for our purposes of 'reasonable baseline'.

P_CORRUPTION = 0.2


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
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hint", action='store_true')
    args = parser.parse_args()
    turns = []

    history = []

    secret = random.sample(noisy_mm.DATASET, k = 1)[0]

    turn = 0
    # now let's play a game
    while True:
        turn += 1
        guess = input("next guess: ")
        if guess not in noisy_mm.DATASET:
            print("E: invalid move")
            continue
        res, _ = noisy_mm.noisy_score(guess, secret, P_CORRUPTION)
        print(f"I: #{turn} {guess} | {res}")
        if res == "4 0":
            turns.append(turn)
            break
        [b, w] = res.split()
        history.append((guess, int(b), int(w)))
        if args.hint:
            print(f"Support: {len(posterior(noisy_mm.DATASET, history, P_CORRUPTION))}")

if __name__ == "__main__":
    main()