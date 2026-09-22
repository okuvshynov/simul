#!/usr/bin/env python3
"""Information-maximising player for Noisy Mastermind, in numpy.

Standalone: it has its own game rules, belief tracking and two policies, and
depends on nothing else in this repository.

    python3 info_player.py --games 100 --p 0.2        # compare greedy vs info
    python3 info_player.py --games 1 --p 0.2 --show   # print one game per policy

The game
--------
The secret is a 4-digit number with distinct digits, first digit nonzero.
A guess is scored as (black, white): digits in the right place, and digits
present but in the wrong place. Before scoring, each digit of the guess is
independently replaced by a wildcard with probability p; a wildcard matches
nothing. An exact guess is never corrupted and always scores (4, 0).

Belief
------
For a candidate secret s and a guess g, the *true* score (B, W) is fixed.
Corruption thins each count independently, so the observed (b, w) is a pair
of binomials: b ~ Bin(B, 1 - p), w ~ Bin(W, 1 - p). That gives an exact
likelihood for any observation under any candidate, and the posterior over
all 4536 secrets is just a product of those likelihoods, normalised.

Policies
--------
greedy  guess the most probable secret.
info    guess whatever maximises the mutual information between the secret
        and the response (the expected reduction in posterior entropy), plus
        a bonus for the chance of winning outright; see info_policy.
"""

import argparse
import math
import time

import numpy as np

DIGITS = 10
LENGTH = 4
N_CODES = 25  # true score (B, W) packed as 5 * B + W, both in 0..4

# ---------------------------------------------------------------------------
# Secrets and the score table
# ---------------------------------------------------------------------------

SECRETS = [str(d) for d in range(1000, 10000) if len(set(str(d))) == LENGTH]
N = len(SECRETS)
INDEX = {s: i for i, s in enumerate(SECRETS)}


def build_code_table():
    """CODE[g, s] = 5 * B + W, the uncorrupted score of guess g against secret s.

    Computed for all pairs at once: B counts equal digits per position, and
    the number of shared digits (B + W) is a dot product of digit-presence
    vectors. 4536 x 4536 bytes, about 20 MB.
    """
    digits = np.array([[int(c) for c in s] for s in SECRETS], dtype=np.int8)     # (N, 4)
    has = np.zeros((N, DIGITS), dtype=np.int32)                                   # has[s, d] = digit d in s
    np.put_along_axis(has, digits.astype(np.int64), 1, axis=1)

    black = np.zeros((N, N), dtype=np.int32)
    for i in range(LENGTH):
        black += digits[:, i][:, None] == digits[:, i][None, :]
    shared = has @ has.T                                                          # B + W
    return (5 * black + (shared - black)).astype(np.uint8)


CODE = build_code_table()


def code_of(black, white):
    return 5 * black + white


# ---------------------------------------------------------------------------
# The noisy channel
# ---------------------------------------------------------------------------

def binomial_pmf(n, q):
    """P(k successes out of n) for k = 0..4, success probability q."""
    return np.array([math.comb(n, k) * q**k * (1 - q) ** (n - k) if k <= n else 0.0 for k in range(5)])


def response_table(p):
    """RESP[c, o] = P(observe code o | true code c) under corruption rate p.

    Row c is the outer product of the two binomial thinnings of B and W.
    Also returns HCOND[c], the entropy of row c, needed for mutual information.
    """
    resp = np.zeros((N_CODES, N_CODES))
    for B in range(5):
        for W in range(5 - B):
            row = np.outer(binomial_pmf(B, 1 - p), binomial_pmf(W, 1 - p))       # (b, w) -> prob
            resp[code_of(B, W)] = row.reshape(-1)[[code_of(b, w) for b in range(5) for w in range(5)]]
    hcond = entropy_rows(resp)
    return resp, hcond


def entropy_rows(dist):
    """Shannon entropy (nats) of each row of a matrix of distributions."""
    safe = np.where(dist > 0, dist, 1.0)
    return -(dist * np.log(safe)).sum(axis=1)


class Channel:
    """The codemaker: corrupts a guess, then scores it. Owns its RNG so a game
    is reproducible from a seed."""

    def __init__(self, p, rng):
        self.p, self.rng = p, rng

    def respond(self, guess, secret):
        if guess == secret:
            return 4, 0                      # an exact guess is never corrupted
        corrupted = "".join("?" if self.rng.random() < self.p else c for c in guess)
        black = sum(g == s for g, s in zip(corrupted, secret))
        white = len(set(corrupted) & set(secret)) - black
        return black, white


# ---------------------------------------------------------------------------
# Belief: posterior over all secrets
# ---------------------------------------------------------------------------

class Belief:
    """Posterior weights over SECRETS; zero for eliminated candidates."""

    def __init__(self, p):
        self.resp, self.hcond = response_table(p)
        self.w = np.full(N, 1.0 / N)

    def update(self, guess_idx, observed_code):
        """Bayes update after guess g produced observed (b, w)."""
        like = self.resp[CODE[guess_idx], observed_code]       # P(obs | each candidate)
        like[guess_idx] = 0.0                                  # the guess itself would have scored (4, 0)
        self.w *= like
        self.w /= self.w.sum()

    def support(self):
        return np.flatnonzero(self.w)

    def entropy(self):
        return entropy_rows(self.w[None, :])[0]

    def mutual_information(self):
        """MI[g] between the secret and the response to guess g, for every g.

        hist[g, c] is the posterior mass on candidates whose true score against
        g is c. The observed-response distribution is hist @ RESP, and
            MI = H(response) - E_secret[H(response | secret)]
        where the second term is hist @ HCOND.
        """
        sup = self.support()
        w = self.w[sup]
        codes = CODE[:, sup]                                    # (N, |support|)
        hist = np.empty((N, N_CODES))
        for c in range(N_CODES):
            hist[:, c] = (codes == c) @ w
        mixture = hist @ self.resp
        return entropy_rows(mixture) - hist @ self.hcond


# ---------------------------------------------------------------------------
# Policies: each maps a belief to a guess index
# ---------------------------------------------------------------------------

def greedy_policy(belief, rng, **_):
    """Most probable secret, ties broken at random."""
    best = np.flatnonzero(belief.w == belief.w.max())
    return int(rng.choice(best))


def info_policy(belief, _rng, win_bonus=math.log(3), **_):
    """Maximise MI(g) + win_bonus * P(g).

    Pure information gain undervalues guesses that can end the game. If the
    turns still needed from a state grow linearly with its entropy, at k turns
    per nat, then after guessing g the expected cost is
        (1 - P(g)) * 1  +  k * (H - MI(g))
    (the game is over with probability P(g); otherwise one turn was spent and
    the expected entropy left is H - MI(g)). Minimising it means maximising
    MI(g) + P(g) / k. Empirically a solver needs about one extra turn per
    tripling of the candidate count, so 1 / k = ln 3 is the default.
    """
    sup = belief.support()
    if len(sup) == 1:                 # nothing left to learn: no guess carries information
        return int(sup[0])
    score = belief.mutual_information() + win_bonus * belief.w
    return int(np.argmax(score))


POLICIES = {"greedy": greedy_policy, "info": info_policy}

# The information score of the very first guess is the same in every game
# (uniform belief), so compute it once.
_OPENING = {}


def choose(policy_name, belief, rng, history_len, **kw):
    if history_len == 0 and policy_name == "info":
        key = kw.get("win_bonus")
        if key not in _OPENING:
            _OPENING[key] = info_policy(belief, rng, **kw)
        return _OPENING[key]
    return POLICIES[policy_name](belief, rng, **kw)


# ---------------------------------------------------------------------------
# Playing games
# ---------------------------------------------------------------------------

def play(secret, policy_name, p, rng, show=False, max_turns=100, **kw):
    """Play one game to completion. Returns the number of guesses used."""
    belief = Belief(p)
    channel = Channel(p, rng)
    for turn in range(1, max_turns + 1):
        g = choose(policy_name, belief, rng, turn - 1, **kw)
        guess = SECRETS[g]
        b, w = channel.respond(guess, secret)
        if show:
            sup = belief.support()
            print(f"  #{turn:2} {guess}  ->  {b} {w}   "
                  f"P(guess)={belief.w[g]:.3f}  support={len(sup)}  H={belief.entropy():.2f} nats")
        if (b, w) == (4, 0):
            return turn
        belief.update(g, code_of(b, w))
    return max_turns


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--p", type=float, default=0.2, help="corruption probability per digit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--win-bonus", type=float, default=math.log(3),
                    help="info policy: nats of information a sure win is worth (default ln 3)")
    ap.add_argument("--policies", default="greedy,info")
    ap.add_argument("--show", action="store_true", help="print every move")
    args = ap.parse_args()

    # same secrets for every policy; corruption draws differ per game but are seeded
    secrets = np.random.default_rng(args.seed).choice(SECRETS, size=args.games, replace=True)

    for name in args.policies.split(","):
        rng = np.random.default_rng(args.seed + 1)
        t0 = time.time()
        turns = []
        for secret in secrets:
            if args.show:
                print(f"{name}: secret {secret}")
            turns.append(play(str(secret), name, args.p, rng, show=args.show, win_bonus=args.win_bonus))
        turns = np.array(turns)
        se = turns.std(ddof=1) / math.sqrt(len(turns)) if len(turns) > 1 else 0.0
        print(f"{name:7} p={args.p}  games={args.games}  mean turns={turns.mean():.2f} ± {se:.2f}  "
              f"max={turns.max()}  ({(time.time() - t0) / args.games:.2f}s/game)")


if __name__ == "__main__":
    main()
