# Equal-GPU crossover measurements

`crossover` compares requests meeting TTFT and mean inter-token-latency targets,
divided by the deployment's total GPU count. It generalizes the dense/MoE
poster experiment to explicit endpoints and equal GPU allocations, using
Python's standard library.

The protocol comes from [poster issue #1](https://github.com/stmcgovern/PTC26-llm-d-diagnostics-poster/issues/1).

## Configure

Copy [the example](../examples/crossover.json). Declare every model instance,
its GPU count, role, and direct `/metrics` URL. A TP2 instance counts as two
GPUs. P/D includes both prefill and decode GPUs. Validation rejects unequal
totals, missing serving targets, and shared metrics URLs.

Deploy the same model revision, weight/KV dtype and relevant serving flags.
Record them in `model_revision` and `deployment_settings`. GPU counts are
declarations: independently verify pod GPU claims, TP and placement. This
command does not deploy or resize model pods.

Targets explicitly select monolithic endpoints or decode sidecars. Each P/D
target carries `x-prefiller-host-port`. Repeat a decode target with different
prefiller headers to describe multiple pairings. This controls manual routing;
EPP routing is outside this command's scope.

Use individual metrics endpoints and reserve them for the experiment. Checks
reject missing telemetry, observed counter resets, cache hits, and unexpected
compute/external-KV token counts. Export pod manifests, startup logs and actual
transport evidence alongside published runs.

## Run

The server must expose vLLM-compatible `/tokenize`, `/v1/completions`, token IDs,
usage, `ignore_eos`, `cache_salt` and prompt-token-source metrics. These APIs
were used by the vLLM v0.31.0 poster deployments.

```bash
python3 toolkit/crossover.py validate examples/crossover.json
python3 toolkit/crossover.py run my-crossover.json --output data/crossover-run
./toolkit/run.sh clusters/my-cluster crossover my-crossover.json
```

For sequential deployments under quota, keep the complete arm declaration
and set one shared `nonce` in the JSON before running either arm. An omitted
nonce is generated independently for each invocation and will not pair:

```bash
./toolkit/run.sh clusters/my-cluster crossover my-crossover.json --arms mono
./toolkit/run.sh clusters/my-cluster crossover my-crossover.json --arms pd
python3 toolkit/crossover.py analyze data/mono/results data/pd/results \
    --output data/crossover-analysis
```

The pod wrapper prints actual result paths, uses a unique remote directory,
copies failed runs too, and verifies checksums before removing its remote files.
Existing result directories are never overwritten.

## Measurements

- Exact prompt token IDs are prepared before timing and retained. A fresh run
  nonce separates invocations; cell keys include length, concurrency and repeat.
  Set the same nonce to pair sequential arm runs. Request-specific cache salts
  keep those identical prompts from reusing KV across arms or requests.
- Unique warmups are retained separately. Their metric publication is checked
  before starting measurement; initially absent source counters are identified
  explicitly during warmup initialization.
- TTFT ends at the first token-bearing event. Mean ITL ends at the last token,
  excluding stream cleanup. Multi-token chunks are retained; per-token p99 is
  unavailable for them. If the first event bundles tokens, mean ITL is unavailable
  and that request cannot qualify for SLO goodput.
- Every request, including failures, is retained. HTTP 200 is insufficient:
  require completed streams, exact usage and exact output token-ID counts.
- Load is a finite closed loop including ramp and drain. Requests must be at
  least three times the largest concurrency. No requests are dropped to infer
  steady state. Longer cells reduce boundary effects.
- Cell order is randomized from the recorded nonce. Repeats are supported;
  analysis does not claim confidence intervals from correlated requests. A
  small observed gain is a sampled result, not an established threshold.

Analysis checks manifests, recomputes SLO counts and GPU-normalized goodput,
checks saved token-ID prompts and source-counter accounting,
rejects duplicate/incompatible cells, and writes `cells.json`,
`comparisons.json` and `comparisons.csv`. All sampled comparisons and later
reversals are preserved. Unmatched cells do not imply a crossover.

Original poster data can be read explicitly with `analyze --legacy-poster`.
Old completion markers/counts are checked; this does not retrospectively
certify the new runner or create original-data manifests. Keep these inputs
identified as legacy.

## Verification

This port was checked with real HTTP/SSE synthetic endpoints locally and in
an OpenShift CPU client pod, including artifact retrieval and failed-run
retention. Reanalysis checked 84 retained GPU cells and 56 comparisons from
the poster campaign. The existing regression suite passed all 474 tests;
lint, type checks, and shell syntax checks passed. These checks did not
constitute a fresh GPU campaign with the new runner.
