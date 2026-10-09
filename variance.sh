jq -rn '[
    "model",
    "effort",
    "n_games",
    "n_solved",
    "n_tokens"
], (inputs | [
    .args.model,
    .args.reasoning_effort,
    .args.n_games,
    .result.n_solved,
    .result.n_tokens_out]
) | @csv' eg_logs/*.json \
| xan filter 'model eq "Qwen3.8-27B-Q8"' 

