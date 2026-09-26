# noisy-mastermind

A task/harness to benchmark reasoning settings of LLMs.

More details in the post: 

Files:

* run.py - main file, running the benchmark
* greedy_player.py - greedy player baseline
* human_cli.py - cli interface to play a game; useful to get a sense of the complexity
* noisy_mm.py - library with common definitions for the puzzle
* run_queue.py - a basic queue for distributed processing across multiple llama.cpp instances.

Scripts:

* boxplot.py     - util to plot results
* budget_hits.py - how often did we go over budget 