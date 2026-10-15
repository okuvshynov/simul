jq -rn '["model", "path", "n_games", "n_solved", "n_tokens"], (inputs | [.args.model, .paths.[0], .args.n_games, .result.n_solved, .result.n_tokens_out]) | @csv' logs/*.json | xan filter 'path ne ""'
