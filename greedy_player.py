import argparse
import math
import random

# Greedy here means 'always going for the potential secret'.
# The sampling of the guess itself can be controlled with temperature.
# This 'greedy' is to contrast with information-gain maximization algorithm.
# Greedy is ok for our purposes of 'reasonable baseline'.

P_CORRUPTION = 0.2

# exact score w/o corruption.
# we assume that both guess and secret are valid
def exact_score(guess: str, secret: str):
    black = sum(a == b for a, b in zip(guess, secret))
    white = len(set(guess).intersection(secret)) - black
    return black, white

# how likely to observe (b, w) if
# true answer before corruption would be (b_, w_)
# and p(corruption) = p
# corruption is independent for each digit
def likelihood(b, w, b_, w_, p):
    # impossible to observe more matches than uncorrupted;
    # corruption can only reduce b & w.
    if b > b_ or w > w_:
        return 0.0
    pb = math.comb(b_, b) * ((1 - p) ** b) * (p ** (b_ - b))
    pw = math.comb(w_, w) * ((1 - p) ** w) * (p ** (w_ - w))

    return pb * pw

# dataset: all valid secrets
# history: [(guess, b, w)] after corruption
# p: probability of corruption for a digit
def posterior(dataset, history, p):
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

        l = 1.0
        for g, b, w in history:
            if g == s:
                if b == 4 and w == 0:
                    return {s: 1.0}
                else:
                    l = 0.0
            else:
                b_, w_ = exact_score(g, s)
                l *= likelihood(b, w, b_, w_, p)

            if l == 0.0:
                break

        if l > 0.0:
            res[s] = l

    norm = sum(res.values())

    # should never happen
    if norm == 0.0:
        raise ValueError("Inconsistency in the history")

    return {s: v / norm for s, v in res.items()}

def score_guess(guess, secret, p_corr=P_CORRUPTION):
    if guess == secret:
        # no corruption if guessed correctly;
        return "4 0", guess

    # do corruption
    guess = "".join('?' if random.random() < p_corr else c for c in guess)
    
    black  = sum(a == b for a, b in zip(guess, secret))
    white  = len(set(guess).intersection(secret)) - black
    result = f"{black} {white}"

    return result, guess

def sample(probs, temp=1.0):
    if temp <= 0.0:
        # greedy
        return max(range(len(probs)), key=lambda k: probs[k])

    logits    = [math.log(p) / temp for p in probs]
    max_logit = max(logits)
    weights   = [math.exp(logit - max_logit) for logit in logits]

    return random.choices(range(len(weights)), weights, k=1)[0]
    
# what we need to do here:
# study how sensitive is it to 'single good turn' vs 'single bad turn'.
# Let's say we play a game and somewhere in midgame one turn is replaced with
# random turn. How much worse will it be?

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--secret")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--quiet", "-q", action='store_true')
    parser.add_argument("--opening", "-o", action='store_true')
    args = parser.parse_args()

    DATASET = [str(d) for d in range(1000, 10000) if len(set(str(d))) == 4]

    secret = args.secret

    turns = []

    for i in range(args.n_samples):
        history = []
        turn = 0
        done = False

        # hardcoded opening
        if args.opening:
            turn += 1
            g0 = "1234"
            res, cg0 = score_guess(g0, secret)
            if not args.quiet:
                print(f"I: #{turn} g({g0} -> {cg0}, {secret}) = {res}")
            if res == "4 0":
                turns.append(turn)
                done = True
            else:
                turn += 1
                g1 = "5678"
                res, cg1 = score_guess(g1, secret)
                if not args.quiet:
                    print(f"I: #{turn} g({g1} -> {cg1}, {secret}) = {res}")
                if res == "4 0":
                    turns.append(turn)
                    done = True

        # now let's play a game
        while not done:
            turn += 1
            belief = list(posterior(DATASET, history, P_CORRUPTION).items())
            probs  = [p for _, p in belief]
            # this player never 'probes' - always choosing one of the 'possible'
            # values, which is suboptimal, but acceptable for this use-case.

            guess, _ = belief[sample(probs, args.temp)]
            res, corrupted_guess = score_guess(guess, secret)
            if not args.quiet:
                print(f"I: #{turn} g({guess} -> {corrupted_guess}, {secret}) = {res}")
            if res == "4 0":
                turns.append(turn)
                break
            [b, w] = res.split()
            history.append((guess, int(b), int(w)))

        if (i + 1) % 100 == 0:
            print(f"I: sample={i + 1} avg n_turns = {sum(turns) / len(turns)}")

if __name__ == "__main__":
    main()