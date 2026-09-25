# Per variant (model-reasoning_effort[-note]), same grouping as boxplots.py.
# Turns/tokens stats are over solved games only; quartiles use linear interpolation.
jq -rs '
  def variant: [.model, .reasoning_effort] + (if .note then [.note] else [] end) | join("-");
  def q(p): sort | ((length - 1) * p) as $i | ($i | floor) as $lo
    | if $lo + 1 < length then .[$lo] + (.[$lo + 1] - .[$lo]) * ($i - $lo) else .[$lo] end;
  def box: "\(q(0.25) | round)/\(q(0.5) | round)/\(q(0.75) | round)";

  (["variant","solved","rate","turns_q1/med/q3","tokens_q1/med/q3","failures"] | @tsv),
  (map(. + {variant: variant}) | group_by(.variant)[]
   | map(select(.status == "solved")) as $s
   | [.[0].variant,
      "\($s | length)/\(length)",
      "\(100 * ($s | length) / length | round)%",
      ($s | map(.n_turns) | box),
      ($s | map(.n_tokens_out_total) | box),
      (map(select(.status != "solved") | .status) | group_by(.) | map("\(.[0])=\(length)") | join(",")
       | if . == "" then "-" else . end)]
   | @tsv)
' logs/*.json | column -t
