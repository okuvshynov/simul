# noisy mastermind

A benchmark to test reasoning settings across quants.

We ask model to solve well known puzzle (mastermind-like), with a quirk where we might corrupt some of of the digits before evaluation, thus model might see lossy results. 

The goal is a little less about comparing models to each other and more about studying their reasoning behaviour. Focus is on testing model in a very simple environment.

For now openai API and local llama.cpp server are verified to work.

We ask model to optimize for average number of turns. Not 'minimal guaranteed number of turns' or 'number of tokens'. 

TODO:

* [x] llama.cpp works with responses API, migrate;
* [ ] log visualizer script
* [ ] timeout handling
* [ ] reasoning cutoff + tool call
* [ ] multi-puzzle conversations; Simultaneous work on multiple puzzles for long context; [v2]
* [ ] grandmaster simul mode: play multiple puzzles at the same time, interleaving the games [v2]