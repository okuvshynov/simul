import argparse

import json
import math
import noisy_mm
import random
import secrets

from greedy_player import posterior, sample_move
from pprint import pprint
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    args = parser.parse_args()

    with open(args.trace, 'r') as file:
        trace = json.load(file)

    for i, p in enumerate(trace["paths"]):
        with open(p, 'r') as file:
            sample = json.load(file)
        history = []
        for turn in sample['turns'][:-1]:
            [b, w] = turn['reply'].split()
            history.append((turn['guess'], int(b), int(w)))
        probs = posterior(noisy_mm.DATASET, history, sample["p_corr"])
        best = max(p for _, p in probs.items())

        guess = trace['result']['guesses'][i]['guess']
        solved = 1 if (guess == sample['secret']) else 0
        model = trace['args']['model']

        p = probs.get(guess, 0.0)
        regret = best - p

        print(f'{solved} {model}: {guess}, p({guess}) = {p:.5} | {regret:.5}')
                

if __name__ == '__main__':
    main()
