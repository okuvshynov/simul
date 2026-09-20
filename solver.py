from math import comb

## this is not exactly solver, more like explorer to produce 
# 'realistic rollouts'.



# exact score w/o corruption.
# we assume that both guess and secret are valid
def score(guess, secret):
    black = sum(a == b for a, b in zip(guess, secret))
    white = len(set(guess).intersection(secret)) - black
    return black, white

# how likely to observe (b, w) if
# true answer before corruption would be (b_, w_)
# and p(corruption) is p (independent for each pos)
def likelihood(b, w, b_, w_, p):
    # impossible to observe more matches than uncorrupted
    # corruption can only reduce
    if b > b_ or w > w_:
        return 0.0
    pb = comb(b_, b) * ((1 - p) ** b) * (p ** (b_ - b))
    pw = comb(w_, w) * ((1 - p) ** w) * (p ** (w_ - w))

    return pb * pw

# dataset: all valid secrets
# history: [(guess, b, w)] after corruption
# p: probability of corruption
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
                b_, w_ = score(g, s)
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

if __name__ == "__main__":
    # manual tests
    print(score("1234", "1234"))
    print(score("1234", "1243"))
    print(score("1234", "4321"))
    print(score("1234", "1290"))

    print(likelihood(3, 0, 3, 1, 0.2))
    print(likelihood(3, 2, 3, 1, 0.2))
    print(likelihood(2, 2, 2, 2, 0.2))
    print(likelihood(0, 0, 2, 2, 0.2))

    DATASET = [str(d) for d in range(1000, 10000) if len(set(str(d))) == 4]
    P_CORRUPTION = 0.2

    belief = posterior(DATASET, [("1234", 2, 0), ("5678", 0, 0)], P_CORRUPTION)
    print(sum(belief.values()))
    print(sorted(belief.items(), key=lambda v: v[1])[-20:])

    belief = posterior(DATASET, [("1234", 2, 1)], P_CORRUPTION)
    print(sum(belief.values()), len(belief))
    print(sorted(belief.items(), key=lambda v: v[1])[-20:])