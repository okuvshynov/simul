# noisy-bench

A benchmark to test reasoning settings across quants.

We ask model to solve well known puzzle (mastermind-like), with a quirk where we might corrupt some of of the digits before evaluation.

TODO:

* [x] llama.cpp works with responses API, migrate;
* [ ] multi-puzzle conversations; Simultaneous work on multiple puzzles for long context; [v2]

Some notes/todos:

* we ask model to optimize for average number of turns. Not 'minimal guaranteed number of turns' or 'number of tokens'. 
* we allow multiple guesses per turn. This is suboptimal from 'optimize number of guesses' point of view, yet models do that;
* 