import argparse

import json
import noisy_mm
import random
import secrets

from greedy_player import posterior, sample_move
from pprint import pprint
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--p_corr", type=float, default=0.2, help="probability of each digit corruption")
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
        probs = posterior(noisy_mm.DATASET, history, args.p_corr)
        #pprint(probs)

        guess = trace['result']['guesses'][i]['guess']
        solved = (guess == sample['secret'])
        #print(guess)
        model = trace['args']['model']

        p = probs.get(guess, 0.0)

        print(f'{solved} {model}: {guess}, p({guess}) = {p}')
                

if __name__ == '__main__':
    main()
