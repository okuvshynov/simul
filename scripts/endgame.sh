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
| xan groupby model,effort,n_games \
"avg(100.0 * n_solved / n_games) as pct_solved, avg(n_tokens) as n_tokens, sum(1) as n_sessions" \
    | xan sort -N -s n_games | xan sort -s model,effort | xan transform pct_solved,n_tokens 'to_fixed(_, 2)' | xan to md

