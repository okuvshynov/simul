import math
import random

import noisy_mm

# Greedy here means 'always going after the potential secret'.
# The sampling of the guess itself can be controlled with temperature.
# This 'greedy' is to contrast with information-gain maximization algorithm.
# Greedy is ok for our purposes of 'reasonable baseline'.

P_CORRUPTION = 0.2
    
def main():
    turns = []

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

if __name__ == "__main__":
    main()