jq -r '[.model, .args.n_games, (.solved | length), .n_turns, .n_tokens_out_total] | @csv' logs/*.json
