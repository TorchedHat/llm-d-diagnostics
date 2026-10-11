# Prefill cost and KV transfer bandwidth

The `bandwidth` command compares direct prefill cost with NIXL transfer cost
over exact prompt lengths, then measures aggregate payload delivery at
configured concurrency. It uses Python 3.10+ and the standard library.
The deployment must already be running; this command creates no GPU pods.

The protocol comes from [poster issue #2](https://github.com/stmcgovern/PTC26-llm-d-diagnostics-poster/issues/2).

## Configure and run

Copy [`examples/bandwidth.json`](../examples/bandwidth.json) and replace its
endpoints and deployment settings. The example represents two nodes with
two GPUs each: one TP2 prefill engine and one TP2 decode engine. Each TP
engine must fit within one node for clocks in its rank traces to be comparable.
The backend must support exact token-ID completion prompts, `cache_salt`,
`return_token_ids`, NIXL telemetry, and source-labelled prompt counters.
Configure the server's context limit for the largest prompt plus one output token.

Use direct per-engine metrics endpoints, including decode's backend port
rather than its routing sidecar. Stop other traffic to both engines.
The supplied `x-prefiller-host-port` header selects manual sidecar routing.
Run EPP performance experiments separately; this is a transfer measurement.
Declare model revision, image digest, GPU type, KV dtype, block size, placement,
and connector settings in `deployment_settings` for reproducibility.

```bash
python3 toolkit/bandwidth.py validate my-bandwidth.json

# In the existing test-client pod; env.sh supplies NS and DATA_DIR.
# Set TEST_CLIENT if the pod is not named test-client.
./toolkit/run.sh clusters/my-cluster bandwidth my-bandwidth.json

# Or directly from a machine that can reach all configured endpoints:
python3 toolkit/bandwidth.py run my-bandwidth.json --output data/bandwidth-run
python3 toolkit/bandwidth.py analyze data/bandwidth-run --output data/bandwidth-analysis
```

Set `EXPERIMENT_API_KEY` in the local environment if authentication is required.
Credentials are not saved in config files. A run uses a new output directory
and nonce; set `nonce` explicitly only when intentionally pairing campaigns.

Serial prefill and transfer phases use identical token-ID prompts with
different cache salts. Warmups and aggregate cells use separate prompts and
salts. Warmups are excluded from accounting. Each request produces exactly
one token, so these runs measure prefill/transfer behavior, not decode throughput.
For hybrid connectors that prefill one fewer token before transfer, set
`prefill_backoff_tokens: 1`; the analysis records this baseline difference.

## TP rank tracing

TP > 1 requires `rank_trace_url`. Backend transfer histograms contain an
observation per rank; adding their durations is not a request's transfer
latency. The runner correlates the response request ID with exactly one
transfer per rank, sums their bytes, and measures the interval from the first
post to the last observed completion on the decode node's common clock.
That interval includes completion observation delay. Rank duration sums,
maximum duration, and start skew are retained separately.

For the vLLM 0.31.0 NIXL source used in the poster experiments,
[`toolkit/nixl_trace.py`](../toolkit/nixl_trace.py) provides optional instrumentation.
It checks both installed source hashes and compiles both patches before
writing either file, and refuses unrecognized versions. Use it only in a
disposable serving container with a writable vLLM installation. Other vLLM
versions need reviewed instrumentation or a compatible trace provider.

Mount this helper into the decode server and a CPU sidecar. Share an
`emptyDir` volume at `/nixl-trace`, set `NIXL_TRACE_DIR=/nixl-trace` in both,
and expose port 9109 only to the experiment client:

```bash
# Decode server startup; preserve your existing vLLM arguments.
python3 /diagnostics/nixl_trace.py instrument -- vllm serve MODEL ...

# CPU sidecar startup; use a Python 3.10+ image.
python3 /diagnostics/nixl_trace.py serve --host 0.0.0.0 --port 9109
```

Start with an empty trace volume when recreating an engine. Do not instrument
the same installation twice. The helper records kernel boot IDs as clock
domains; differing rank clocks, resets, missing ranks, unrelated transfers,
and trace/Prometheus discrepancies invalidate the cell.

A compatible trace provider exposes GET `/snapshot`. Its JSON contains
`counts` (file name to complete-record offset) and `rows`. Query parameters
contain those offsets; the response includes only later rows and current
counts. Each row supplies `rank`, `request_id`, `clock_domain`,
`posted_monotonic_ns`, `observed_done_monotonic_ns`, `bytes`, `nixl_xfer_us`,
and `nixl_post_us`. Request IDs must match the response ID or start with that
ID plus `-`. All ranks must share a clock domain. TP1 can use backend
telemetry without tracing; the output labels that different timing scope.

## Checks and interpretation

The runner waits for metric publication, verifies request and TP transfer
counts, rejects observed resets, checks error counter deltas and prompt token
sources, and requires zero prefix-cache reuse. Metric names match exactly;
Prometheus `_created` timestamps are not treated as errors. Failed requests
are retained and invalidate the run without replacement or retry. Before/after
metrics, per-request measurements, prompts, ranks, failures, config, source
hashes, and checksums are saved. The pod wrapper copies failed artifacts too.

Optional `kv_bytes_per_token` (total across ranks), `block_size`, and
`state_bytes_per_request` verify a declared payload model:
`ceil(prefill_tokens / block_size) * block_size * kv_bytes_per_token + state_bytes_per_request`.
Use the actual connector layout, including hybrid state. Omit the geometry
fields if that layout cannot be represented by this formula; measured bytes
remain available. The example leaves geometry unspecified.

Analysis recomputes measurements from raw counters and traces before reporting:

- Transfer/prefill ratio with a paired request bootstrap interval. This interval
  describes variation within a serial cell, not variation across deployments or days.
- Payload-only bandwidth thresholds for transfer to take 10% or 100% of direct
  prefill time. These omit fixed overhead and are not a network provisioning guarantee.
- A descriptive payload/time fit with residuals and a flag for nonphysical
  intercept/slope. Optional `reported_port_GB_s` flags a fit above a declared
  port rate; this does not validate NIC wire bandwidth.
- Aggregate backend payload divided by the full request-loop wall time,
  including inference. This does not measure link saturation.
- Direct-versus-PD output token disagreements, retained without filtering.

`transport` is a declared configuration/control label. Raw UCX logs or NIC
counter evidence can be referenced in `transport_evidence`; the analyzer does
not certify RDMA from a label or payload rate. Run RDMA, TCP, and same-node
controls with separate labels and identical model settings. Local loopback
results do not establish cross-node link performance.

Existing poster issue #2 campaigns can be checked with
`analyze --legacy-poster CAMPAIGN... --output NEW_DIR`. This verifies their
saved counters, request usage, paired prompt hashes, and rank traces, while
marking their older provenance format. Witness cells are excluded. New runs
require completion markers and valid manifests.

## Verification

This port was checked with real HTTP/SSE synthetic endpoints locally and in
an OpenShift CPU client pod, including TP2 rank correlation, lazy metric
registration, artifact retrieval, and failed-run retention. Reanalysis checked
104 retained GPU cells and 40 paired length comparisons from the poster
campaigns. The tracing patches were checked against the official vLLM 0.31.0
source hashes; their HTTP offset API and refusal of unsupported source were
also checked. Lint, type checks, and shell syntax checks passed. These checks
did not constitute a fresh GPU campaign with the new runner.
