import sys
import math
import random

from pathlib import Path

P_CORRUPTION = 0.2
def score_guess(guess, secret):
    if guess == secret:
        # no corruption if guessed correctly;
        return "4 0", guess

    # do corruption
    guess = "".join('?' if random.random() < P_CORRUPTION else c for c in guess)
    
    black  = sum(a == b for a, b in zip(guess, secret))
    white  = len(set(guess).intersection(secret)) - black
    result = f"{black} {white}"

    return result, guess

# this is 'player' - we'll use it to test our solver
# and eventually use for data generation.

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver  # noqa: E402

def sample(probs, temp=1.0):
    if temp <= 0.0:
        # greedy
        return max(range(len(probs)), key=lambda k: probs[k])

    logits    = [math.log(p) / temp for p in probs]
    max_logit = max(logits)
    weights   = [math.exp(logit - max_logit) for logit in logits]

    return random.choices(range(len(weights)), weights, k=1)[0]
    

def main():
    secret = sys.argv[1]
    DATASET = [str(d) for d in range(1000, 10000) if len(set(str(d))) == 4]
    if secret not in DATASET:
        print(f"E: invalid secret {secret}")
        exit(1)

    history = []
    turn = 0

    # now let's play a game
    while True:
        turn += 1
        belief = list(solver.posterior(DATASET, history, P_CORRUPTION).items())
        probs  = [p for _, p in belief]
        # greedy. this player never 'probes' - always choosing one of the 'possible'
        # values, which is suboptimal. need to figure out how to estimate
        # information gain. Then we can get more diverse trajectories by 
        # randomly choosing information gain vs greedy (+temp).
        guess, _ = belief[sample(probs, 0)]
        res, corrupted_guess = score_guess(guess, secret)
        print(f"I: #{turn} g({guess} -> {corrupted_guess}, {secret}) = {res}")
        if res == "4 0":
            break
        [b, w] = res.split()
        history.append((guess, int(b), int(w)))


if __name__ == "__main__":
    main()