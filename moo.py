import random

P_CORRUPTION = 0.1
MAX_TURNS = 50
DATASET = [str(d) for d in range(1000, 10000) if len(set(str(d))) == 4]

PROMPT = f"""
Let's play a game of Bulls and Cows/Mastermind with imperfect communication channel.

# Rules

Your task is to guess the secret number. The secret number has four distinct digits 0..9 with no repetitions, cannot start with 0.
Use the provided tool 'make_guess'.
The output you get is a string "Bulls Cows".
"4 0" would indicate solved problem.

# Scoring

For every game instance the number of guesses to solve the problem will be recorded.
If you fail to find a solution in {MAX_TURNS} guesses, the puzzle is marked as unsolved.
Your priorities are (in order):

1. solve as many puzzles as possible
2. minimize the average number of guesses for solved ones.


# Restrictions

Your guess must be a potentially valid solution to the puzzle. Do not try to game the system by making a guess with repeated/non-digit characters. You'll get 'error' as tool result and it will be still counted towards your turn count. 
You can make multiple guesses per turn, but all guesses within that turn will be counted toward the score.

# Communication.

If you guessed the correct secret number, you will always get "4 0".

If not, before evaluating your guess, each of the digits in your guess might be corrupted. 
Corruption is independent per digits, p(corruption) for each digit is the same number p = {P_CORRUPTION}.
Corrupted digit is not recoverable and will be ignored when computing bulls/cows.

## Example 1:

secret = 1234
your guess = 1235

after corruption, guess = 1?35

the output you get is "2 0".

## Example 2:

secret = 1234
your guess = 1234

You guessed correctly, so there's no chance of corruption. the output you get is "4 0".

## Example 3:

secret = 1234
your guess = 1236

after corruption:

guess = 1??6

Two digits were corrupted you'll get "1 0".

Your turn.
"""

MAKE_GUESS_FUNCTION = {
    "name" : "make_guess",
    "description" : "Make a guess in a game. The input is four digit guess, the output is a single string with two numbers - bulls & cows respectively, taking into account probabilistic corruption.",
    "parameters" : {
        "type" : "object",
        "properties" : {
            "guess": {
                "type" : "string",
                "description" : "four digit guess for the game. Your guess must be a valid potential solution to the puzzle - 4 digits, no repeated digits, not starting with 0, no other symbols. If it is invalid, you'll get 'error' string as a result."
            }
        },
        "required" : ["guess"]
    }
}

TOOLS = [
    {
        "type" : "function",
        "function" : MAKE_GUESS_FUNCTION
    }
]

TOOLS_OPENAI = [
    {
        "type" : "function",
        **MAKE_GUESS_FUNCTION
    }
]

# returns result + corrupted version
def make_guess(guess, secret):
    if guess == secret:
        # no corruption possible
        return "4 0", guess

    if guess not in DATASET:
        # guess was not a valid candidate
        return "error", guess

    # do corruption
    guess = "".join('?' if random.random() < P_CORRUPTION else c for c in guess)
    
    bulls = sum(1 if a == b else 0 for a, b in zip(guess, secret))
    cows = len(set(guess).intersection(secret)) - bulls
    result = f"{bulls} {cows}"
    return result, guess