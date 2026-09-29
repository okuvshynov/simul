jq -r '[(.status | if has("solved") then "solved:\(.solved | length)" else "error: \(.error)" end), .model, .n_turns, .n_tokens_out_total, .args.n_games] | @csv' logs/*.json
