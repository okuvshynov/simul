jq -rs '
  def variant: [.model, .reasoning_effort] + (if .note then [.note] else [] end) | join("-");
  def avg: if length > 0 and all(. != null) then add / length | round else "---" end;

  (["variant","solved","turns_avg","tokens_avg","failures"] | @tsv),
  (group_by(variant)[]
   | map(select(.status == "solved")) as $s
   | [(.[0] | variant),
      "\($s | length)/\(length)",
      ($s | map(.n_turns) | avg),
      ($s | map(.n_tokens_out_total) | avg),
      (map(select(.status != "solved") | .status) | group_by(.) | map("\(.[0])=\(length)") | join(",")
       | if . == "" then "-" else . end)]
   | @tsv)
' logs/*.json | column -t
