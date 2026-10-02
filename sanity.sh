#!/usr/bin/env bash

jq -rn '[
    "model",
    "effort",
    "n_games",
    "n_solved",
    "n_turns",
    "n_tokens",
    "n_warnings"
], (inputs | [
    .model,
    .args.reasoning_effort,
    .args.n_games,
    (.solved | length),
    .n_turns,
    .n_tokens_out_total,
    ([.trace[].warnings[]?] | length)]
) | @csv' logs/*.json | xan groupby model,effort,n_games "avg(n_turns / n_games) as n_turns, avg(n_tokens / n_games) as n_tokens, avg(100.0 * n_solved / n_games) as pct_solved, sum(1) as n_sessions, avg(n_warnings / n_games) as n_warnings" | xan sort -N -s n_games | xan sort -s model,effort | xan transform n_turns,n_tokens,pct_solved,n_warnings 'to_fixed(_, 2)' | xan to md
