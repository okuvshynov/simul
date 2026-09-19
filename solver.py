from math import comb

def score(guess, secret):
    black = sum(a == b for a, b in zip(guess, secret))
    white = len(set(guess).intersection(secret))
    return black, white

def likelihood(black, white, black_, white_, p_corr):
    