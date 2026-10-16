jq -rn '[
    "model",
    "effort",
    "n_games",
    "n_solved",
    "n_tokens"
], (inputs | [
    .args.model,
    .args.reasoning_effort,
    (.paths | length),
    .result.n_solved,
    .result.n_tokens_out]
) | @csv' logs/*.json

