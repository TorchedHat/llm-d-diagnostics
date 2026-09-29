# llm-d Diagnostics Toolkit

Portable experiment suite for characterizing
[llm-d](https://github.com/llm-d/llm-d) disaggregated inference deployments.
Point it at a cluster, run experiments, get reproducible data.

Python 3.8+, zero external dependencies (stdlib only).

## Architecture

Three layers, two execution domains:

```
┌─────────────────────────────────────────────────────────┐
│  Your machine (or CI)                                   │
│                                                         │
│  run.sh ─── sources env.sh, copies scripts to pod,     │
│             runs experiments via oc exec,               │
│             copies results back                         │
│                                                         │
│  exp4_fault.py ─── runs locally (kills pods via oc)     │
│  analyze.py ────── runs locally (reads CSVs)            │
└──────────────────────┬──────────────────────────────────┘
                       │ oc exec / oc cp
┌──────────────────────▼──────────────────────────────────┐
│  test-client pod (in-cluster)                           │
│                                                         │
│  exp1-3, exp5-7 ──── send requests, measure timing     │
│  fault_driver.py ─── probe + load modes (for exp4)     │
│  metrics_collector.py ── scrape Prometheus /metrics     │
│                                                         │
│  Talks to vLLM pods via in-cluster DNS:                 │
│    vllm-prefill-svc:8100  (prefill, headless)             │
│    vllm-decode-svc:8000   (decode, via sidecar, headless)│
│    vllm-decode-direct-svc:8001 (decode, bypass sidecar)  │
└─────────────────────────────────────────────────────────┘
```

**Why this split?** Experiments 1-7 measure latency and throughput — they
need to run in-cluster to avoid measuring `oc exec` round-trip overhead.
Experiment 4 kills pods — it must run outside the cluster (you can't
`oc delete` from inside a pod without a service account with elevated
privileges).

## Running Experiments

### Prerequisites

1. A running llm-d deployment (prefill + decode pods)
2. A `test-client` pod in the same namespace (see `manifests/`)
3. A cluster config file (`clusters/<name>/env.sh`)

### Quick start

```bash
# Full performance characterization (preflight + all experiments + analysis)
./toolkit/run.sh clusters/my-cluster characterize

# Just check the cluster is ready
./toolkit/run.sh clusters/my-cluster preflight

# Run a single experiment
./toolkit/run.sh clusters/my-cluster latency
./toolkit/run.sh clusters/my-cluster decompose
./toolkit/run.sh clusters/my-cluster throughput

# Fault tolerance assessment (destructive — kills pods)
./toolkit/run.sh clusters/my-cluster fault-test

# Run specific fault sub-experiments
./toolkit/run.sh clusters/my-cluster fault 4h 4k --skip-control

# Re-run analysis on existing data
./toolkit/run.sh clusters/my-cluster analyze
```

### Standalone metrics collection

```bash
# Scrape vLLM /metrics for 60 seconds
COLLECT_DURATION=60 ./toolkit/run.sh clusters/my-cluster metrics
```

## Experiments

### Performance (run in-cluster via test-client pod)

| Command | What it measures | Key output |
|---------|------------------|------------|
| `latency` | Per-request TTFT overhead across prompt lengths (10-1000 tokens). Sequential, no concurrency. Compares BASELINE vs DISAGG-D1 vs DISAGG-D2. | `exp1-results.csv` |
| `decompose` | Isolates overhead sources: A) prefill direct, B) decode direct (no sidecar), C) sidecar-only, D) full disagg. Derives T_sidecar, T_prefill_rt, T_overhead. | `exp1b-results.csv` |
| `throughput` | Throughput scaling at concurrency 1-16. Compares BASELINE (1 GPU) vs DISAGG-1D (2 GPU) vs DISAGG-2D (3 GPU). | `exp2-results.csv` |
| `isolation` | Head-of-line blocking: 1 heavy (1000 tokens) + 5 light (10 tokens) simultaneously. Measures whether P/D protects light requests. Streaming TTFT + ITL. | `exp3-results.csv` |
| `seqlen` | Decomposition at multiple prompt lengths to find the crossover where KV transfer time exceeds prefill compute time. | `exp5-results.csv` |
| `saturation` | Open-loop QPS sweep (1-32 QPS) to find where each topology saturates. Detects the knee where p99 exceeds SLO. | `exp6-results.csv` |
| `mixed` | Realistic traffic: 80% short + 20% long, Poisson arrivals, streaming. The bottom-line verdict on disaggregation. | `exp7-results.csv` |

### Fault tolerance (runs locally, kills pods)

| Sub | Name | What it tests |
|-----|------|---------------|
| 4a | Decode failure | Kill decode-1, verify decode-2 unaffected, measure recovery time |
| 4b | Prefill failure | Kill prefill, verify both decode pods detect it, measure recovery |
| 4d | Graceful degradation | Scale decode-2 to 0 (3 GPU → 2 GPU), verify decode-1 unaffected, scale back |
| 4e | Failure under load | Sustained QPS + kill prefill mid-stream, count pre/post-kill successes |
| 4f | Network partition | Apply NetworkPolicy blocking NIXL traffic, verify timeout behavior |
| 4g | Slow network | tc netem delay + packet loss on prefill, measure TTFT degradation |
| 4h | UCX keepalive | Kill prefill, measure kill-to-detection gap via continuous probe. Multiple runs for variance. Clock-skew corrected. |
| 4i | Mid-transfer failure | Kill prefill at configurable delays during KV transfer, verify clean error (not corruption) |
| 4j | Container vs pod restart | Compare recovery: container restart (same IP) vs pod restart (new IP) |
| 4k | Load + failure | Sustained QPS with mid-test prefill kill. Partition results by kill epoch. |
| 4l | Rolling update | Trigger rolling update under sustained load. Zero-downtime test. |

Exp4 includes:
- **Calibration** (Phase 0a): measures `oc exec` overhead, clock skew between local and pod, probe interval jitter. These define the measurement error budget.
- **Baseline** (Phase 0b): 30-request baseline with confidence intervals and stationarity check.
- **Control run** (Phase 0c): probe + load with no faults to isolate measurement artifacts.
- **Steady-state gates**: between each experiment, verify the system returned to baseline before proceeding.
- **Predictions**: Fermi-style predictions before each experiment, compared to measurements after.

```bash
# Full fault tolerance assessment
./toolkit/run.sh clusters/my-cluster fault-test

# Run specific sub-experiments
./toolkit/run.sh clusters/my-cluster fault 4h 4k --skip-control

# Or invoke directly
python3 toolkit/exp4_fault.py 4h --skip-control
python3 toolkit/exp4_fault.py --stop-on-failure
```

## Configuration

All configuration is via environment variables. Nothing is hardcoded to a
specific cluster, model, or namespace.

### Cluster config (env.sh)

```bash
export NS=my-namespace
export MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
export DATA_DIR="clusters/my-cluster/data"
export PREFILL_HOST="vllm-prefill-svc.${NS}.svc.cluster.local:8100"

# For deploy.sh only:
export VLLM_IMAGE="vllm/vllm-openai:v0.26.0"
export SIDECAR_IMAGE="ghcr.io/llm-d/llm-d-router-disagg-sidecar:v0.10.0"
export GPU_ALLOCATION_MODE="dra"
export GPU_COUNT=1
# Use GPU_ALLOCATION_MODE=classic and GPU_RESOURCE_NAME for classic NVIDIA clusters.
# export GPU_RESOURCE_NAME="nvidia.com/gpu"
export SIDECAR_SCHEME="http"
export MODEL_CACHE_SIZE="50Gi"
```

### Experiment tuning (env vars)

These override defaults in `client.py` and the experiment scripts:

| Variable | Default | Used by | Description |
|----------|---------|---------|-------------|
| `WARMUP` | 3 | exp1-3,5-7 | Warmup requests (discarded) |
| `RUNS` | 20 | exp1-3,5-7 | Measured runs per config |
| `MAX_TOKENS` | 20 | all | Max completion tokens |
| `PROMPT_LENGTHS` | 10,50,100,500,1000 | exp1 | Prompt lengths to sweep |
| `CONCURRENCY_LEVELS` | 1,2,4,8,16 | exp2 | Concurrency levels |
| `QPS_LEVELS` | 1,2,4,8,12,16,24,32 | exp6 | QPS levels to sweep |
| `QPS` | 4 | exp7 | Target QPS for mixed workload |
| `DURATION_S` | 60 | exp6,7 | Duration per QPS level |
| `SIM` | (unset) | all | Set to `1` for inference-sim mode |

### Exp4-specific tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `BASELINE_N` | 30 | Baseline sample size |
| `LOAD_QPS` | 4 | Target QPS for load tests (4e, 4k, 4l) |
| `LOAD_DURATION` | 60 | Load test duration in seconds (4k) |
| `KILL_AT_S` | 10 | Seconds before kill in load tests (4k) |
| `ROLLOUT_DURATION` | 90 | Rolling update test duration (4l) |
| `KEEPALIVE_RUNS` | 3 | Number of kill-to-detection measurement runs (4h) |
| `KILL_DELAYS_MS` | 100,200,500 | Kill delays for mid-transfer test (4i) |
| `NETEM_DELAY_MS` | 100 | Added latency for slow-network test (4g) |
| `NETEM_LOSS_PCT` | 10 | Packet loss percentage (4g) |
| `MID_TRANSFER_PROMPT_TOKENS` | 500 | Prompt size for mid-transfer test (4i) |
| `METRICS_INTERVAL` | 2 | Prometheus scrape interval in seconds |

## Output

All data goes to `$DATA_DIR` (set in env.sh, typically `clusters/<name>/data/`).

```
clusters/my-cluster/data/
├── run-info.json              # Toolkit version, config, timestamps
├── exp1-results.csv           # Per-experiment CSV data
├── exp1b-results.csv
├── exp2-results.csv
├── exp3-results.csv
├── exp4-results.csv
├── exp4-logs/                 # Pod logs captured at key moments during faults
│   ├── 4a-decode1-pre-vllm.log
│   ├── 4a-decode1-post-vllm.log
│   ├── 4h-decode1-post-1-vllm.log
│   └── ...
├── exp4-metrics.csv           # Prometheus timeseries (scraped during exp4)
├── exp5-results.csv
├── exp6-results.csv
├── exp7-results.csv
└── exp7-gpu.csv
```

### CSV schema (performance experiments)

All experiments write CSV with at least:

| Column | Description |
|--------|-------------|
| `config` | Configuration name (BASELINE, DISAGG-D1, etc.) |
| `prompt_length` | Prompt token count |
| `ttft_ms` | Time to first token (milliseconds) |
| `total_ms` | Total request time |
| `status_code` | HTTP status code |
| `error` | Error message (empty on success) |

Streaming experiments (exp3, exp7) add: `itl_mean_ms`, `itl_p99_ms`, `token_count`.

### CSV schema (exp4 fault tolerance)

| Column | Description |
|--------|-------------|
| `experiment` | Always `exp4` |
| `sub` | Sub-experiment (4a, 4b, ..., 4l) |
| `phase` | Phase within sub-experiment (pre, kill, during, post, etc.) |
| `epoch_ms` | Timestamp (milliseconds since epoch) |
| `ttft_ms` | TTFT measurement |
| `total_ms` | Total request time |
| `status_code` | HTTP status or "n/a" for control events |
| `detect_epoch_ms` | When failure was triggered or detected |
| `recover_epoch_ms` | When recovery was confirmed |
| `probes_to_detect` | Probe count until failure detected |
| `probes_to_recover` | Probe count until recovery confirmed |

## Analysis

```bash
./toolkit/run.sh clusters/my-cluster analyze
```

The analyzer (`analyze.py`) reads all CSV files from `$DATA_DIR` and produces:

1. **Data quality checks** — non-200 responses, truncated responses, high error rates
2. **Per-experiment statistics** — mean, median, CI, p99, IQR for each config group
3. **Cross-config comparisons** — overhead ratios, isolation ratios, saturation points
4. **Exp4 analysis** — detection gaps, recovery times, log event categorization
5. **Exp7 verdict** — the bottom line: TTFT overhead, ITL stability, goodput

## Measurement Methodology

### TTFT measurement

Uses `http.client` (not urllib/requests) for precise first-byte timing. The
TTFT measures wall-clock time from connection to first response byte. For
non-streaming requests, vLLM buffers the full response, so TTFT ≈ total time.
For streaming (exp3, exp7), TTFT measures true time-to-first-token via SSE.

### In-pod vs out-of-pod

Performance experiments run inside the test-client pod to avoid measuring
`oc exec` overhead (~100-500ms per call). The `fault_driver.py` probe mode
provides ~200ms resolution for fault detection timing, independent of the
`oc` API round-trip.

### Clock domains (exp4)

Exp4 operates in two clock domains: local machine (kill timing) and pod
(probe/load timing). Phase 0a calibrates the skew between them using a
midpoint estimator. All cross-domain comparisons (detection gaps, load
result partitioning) apply or document the clock correction.

### Statistical rigor

- Bessel-corrected sample variance (N-1 denominator)
- 95% confidence intervals (z=2.0 for n<30, z=1.96 for n>=30)
- Split-half stationarity check on baselines
- Steady-state gates between fault experiments (CI-based threshold)
- Error budget propagation (clock skew + probe jitter → detection uncertainty)

## Files

```
toolkit/
├── README.md              # This file
├── run.sh                 # Experiment runner (copies to pod, runs, copies back)
├── client.py              # Shared config, HTTP client, CSV writer
├── analyze.py             # Statistical analysis of CSV results
├── fault_driver.py        # In-pod probe + load generator (for exp4)
├── metrics_collector.py   # In-pod Prometheus scraper
├── exp1_latency.py        # Experiment 1: latency sweep
├── exp1b_decompose.py     # Experiment 1b: latency decomposition
├── exp2_throughput.py      # Experiment 2: throughput under load
├── exp3_isolation.py      # Experiment 3: prefill isolation
├── exp4_fault.py          # Experiment 4: fault tolerance (runs locally)
├── exp5_seqlen_sweep.py   # Experiment 5: sequence length sweep
├── exp6_saturation.py     # Experiment 6: saturation profiling
└── exp7_mixed_workload.py # Experiment 7: mixed workload
```
