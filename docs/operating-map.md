# Operating-map summaries

`toolkit/operating_map.py` recomputes relative goodput winners from per-cell CSV
summaries such as the poster's four-model study. It also reports each arm's observed
joint-SLO attainment, so a relative advantage during overload is visible alongside
its absolute service level. It reads saved data and does not send requests.

```sh
python3 toolkit/operating_map.py cells.csv --config examples/operating-map.json \
  --output operating-map.json
```

The configuration declares the exact arm names, repeat IDs, equal GPU count,
practical margin and an optional observed-attainment gate. `null` keeps only the
historical relative comparison; a numeric gate additionally reports eligible arms
and a gated winner. Choose the gate before evaluating new data. The example reflects
one eight-GPU experiment, not a general recommended setting.

Input CSV rows need `model`, `revision`, `arm`, `phase`, `repeat`, `prompt_tokens`,
`rate_key`, `requests`, `successes`, `failures`, `slo_passes`, `wall_seconds`,
`goodput_per_gpu` and `valid`. For each main-phase workload, every configured arm
and repeat must occur once. The tool checks counts and recomputes finite goodput
from SLO passes, cell wall time and the declared GPU count. A relative winner must
exceed the next arm by the selected margin and lead in every repeat. The output
retains the source hashes and per-cell counts.

The tool consumes summaries: it does not re-audit individual requests, confirm
paired prompts, verify actual GPU allocations or validate quality. Two repeats are
not a confidence interval; finite wall-time goodput includes the measurement's
ramp and drain. An observed attainment gate does not establish a population SLO.
