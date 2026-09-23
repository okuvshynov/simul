import random

DATASET = [str(d) for d in range(1000, 10000) if len(set(str(d))) == 4]

# exact score w/o corruption.
# we assume that both guess and secret are valid
def exact_score(guess: str, secret: str):
    black = sum(a == b for a, b in zip(guess, secret))
    white = len(set(guess).intersection(secret)) - black
    return black, white

def noisy_score(guess, secret, p_corr, rng_noise=None):
    if guess == secret:
        # no corruption if guessed correctly;
        return "4 0", guess

    # do corruption
    if rng_noise is None:
        rng_noise = random.Random()
    guess = "".join('?' if rng_noise.random() < p_corr else c for c in guess)
    
    black  = sum(a == b for a, b in zip(guess, secret))
    white  = len(set(guess).intersection(secret)) - black
    result = f"{black} {white}"

    return result, guess