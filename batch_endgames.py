# util to shuffle and split input list to batches

import random
import sys

def main():
    with open(sys.argv[1], 'r') as f:
        sessions = [s.strip() for s in f.readlines()]

    bsz = int(sys.argv[2])
    while len(sessions) > bsz:
        print(",".join(sessions[:bsz]))
        sessions = sessions[bsz:]

if __name__ == '__main__':
    main()
