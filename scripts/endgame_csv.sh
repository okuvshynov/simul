#!/bin/sh
# Flatten endgame logs into one CSV row per session (input for viz/endgame.html).
# stop: why the response ended early (max_tokens, max_output_tokens, ...), empty if it finished.
# A session with n_tokens == 0 never got a response (API failure).
# Usage: scripts/endgame_csv.sh > out/endgames_viz.csv
cd "$(dirname "$0")/.." || exit 1
jq -rn '[
    "model",
    "effort",
    "n_games",
    "n_solved",
    "n_tokens",
    "stop"
], (inputs | [
    .args.model,
    .args.reasoning_effort,
    (.paths | length),
    .result.n_solved,
    .result.n_tokens_out,
    (.result.incomplete_reason // "")]
) | @csv' logs/*.json
