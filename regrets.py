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
    parser.add_argument("--sample", required=True)
    parser.add_argument("--p_corr", type=float, default=0.2, help="probability of each digit corruption")
    args = parser.parse_args()
    with open(args.sample, 'r') as file:
        sample = json.load(file)

    history = []
    for turn in sample['turns'][:-1]:
        [b, w] = turn['reply'].split()
        history.append((turn['guess'], int(b), int(w)))
    probs = posterior(noisy_mm.DATASET, history, args.p_corr)
    pprint(probs)

    files = sorted(Path("logs").glob("*.json"))
    for f in files:
        with open(f, 'r') as file:
            content = json.load(file)
            if 'paths' not in content:
                continue

            if len(content['paths']) != 1:
                continue

            if content['paths'][0] != args.sample:
                continue

            result = content['result']
            is_solved = result['n_solved']
            guess = result['guesses'][0]['guess']
            model = content['args']['model']

            p = probs.get(guess, 0.0)

            print(f'{model}: {guess}, p({guess}) = {p} | solved={is_solved}')
                

if __name__ == '__main__':
    main()
