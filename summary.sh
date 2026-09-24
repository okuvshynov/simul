jq -rs '
  (["note","status","runs","avg_turns","avg_tokens"] | @tsv),
  (group_by([.note, .status])[]
   | [.[0].note, .[0].status, length,
      (map(.n_turns) | add / length | round),
      (map(.n_tokens_out_total) | add / length | round)]
   | @tsv)
' logs/*.json | column -t