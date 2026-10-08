# llm-d Diagnostics Toolkit

Portable diagnostics for [llm-d](https://github.com/llm-d/llm-d)
disaggregated (prefill/decode) inference deployments. Point it at a
cluster, run experiments, get a performance characterization.

Python 3.8+, zero external dependencies. Runs inside any Kubernetes
cluster via a test-client pod. Requires `oc` (OpenShift CLI) or
`kubectl` with minor script edits.

## Quick Start

```bash
# 1. Create a cluster config
mkdir -p clusters/my-cluster/data
cp examples/env.sh.example clusters/my-cluster/env.sh
# Edit env.sh with your namespace, model, images

# 2. Deploy
# Optional: validate/render resources without writing to the cluster
DRY_RUN=client ./scripts/deploy.sh clusters/my-cluster
# Or, against an existing namespace, ask the API server to validate them:
DRY_RUN=server ./scripts/deploy.sh clusters/my-cluster
./scripts/deploy.sh clusters/my-cluster

# 3. Run all non-destructive experiments
./toolkit/run.sh clusters/my-cluster characterize

# 4. Run a single experiment
./toolkit/run.sh clusters/my-cluster latency
```

Each experiment writes CSV data. The analyzer computes medians,
confidence intervals, and flags data quality issues.

## Experiments

| Command | What it measures |
|---------|-----------------|
| `latency` | Per-request overhead of disaggregation |
| `decompose` | Breakdown: routing sidecar vs NIXL (KV cache transfer) overhead |
| `throughput` | Scaling under concurrent load |
| `isolation` | Whether prefill/decode separation protects light requests from heavy prefills |
| `seqlen` | Transfer cost vs prompt length |
| `saturation` | QPS at which each topology collapses |
| `mixed` | Realistic mixed-length workload comparison |
| `fault` | Pod failure and recovery time (destructive) |
| `prefix-cache` | KV cache hit rates across requests and pods |
| `model-load` | Cold start time (kills pods) |
| `kv-eviction` | KV cache persistence under delay and pressure |

Run `characterize` for all non-destructive experiments. Run `fault-test`
for destructive experiments (kills pods -- confirms before running).

## Advisory Tools

The `advisor/` directory adds decision-making on top of the measurement data.

```bash
# Should I disaggregate? (no cluster needed)
python3 advisor/plan.py --model meta-llama/Llama-3.1-8B-Instruct --gpu-type h100

# What's wrong with my cluster?
./toolkit/run.sh clusters/my-cluster diagnose

# Is my P/D ratio right?
./toolkit/run.sh clusters/my-cluster rebalance
```

Capacity planning, root-cause diagnosis, P/D ratio optimization, and health monitoring. See [advisor/README.md](advisor/README.md) for details.

## What You Get

After `characterize`, the `data/` directory has CSVs for each experiment.
Run the analyzer:

```bash
# analyze after characterize finishes:
./toolkit/run.sh clusters/my-cluster analyze
# or run the analyzer directly:
python3 toolkit/analyze.py clusters/my-cluster/data/
```

The analyzer produces:
- Median latency with 95% confidence intervals
- Statistical comparison between topologies (Mann-Whitney U)
- Outlier detection and data quality flags
- Overhead attribution (sidecar, transfer, decode)

Run `characterize` and `analyze` against your cluster to produce a
complete assessment.

## Simulation Mode

Validate the toolkit without GPUs using
[llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim):

```bash
cp examples/env-sim.sh.example clusters/my-sim/env.sh
./scripts/deploy.sh clusters/my-sim sim
SIM=1 ./toolkit/run.sh clusters/my-sim latency
```

Sim mode exercises the sidecar routing and measurement pipeline with
canned responses. No real model, no KV cache, no NIXL transfer — the
numbers don't reflect real inference.

The GPU deployment defaults use upstream vLLM `v0.31.0` with the llm-d
Router disaggregation sidecar `v0.11.0`. The llm-d v0.10.0 release matrix
lists vLLM `v0.30.0`; use that image when reproducing that exact matrix.
Override `VLLM_IMAGE` for an accelerator-specific image in your cluster. Set
`GPU_ALLOCATION_MODE=dra` on OpenShift DRA clusters; it requests the existing
`gpu-<GPU_COUNT>` ResourceClaimTemplate for every vLLM pod. Use
`GPU_ALLOCATION_MODE=classic` and `GPU_RESOURCE_NAME` on clusters with the
classic NVIDIA device plugin.

## Profiling Tools

When the diagnostics show something unexpected, the `profiling/`
directory has PyTorch ecosystem wrappers to investigate *why*:

```bash
# Apply debug env vars to a deployment
python3 profiling/env_presets.py apply flight-recorder vllm-decode -n mynamespace

# Generate a profiling wrapper for vLLM
python3 profiling/torch_profiler.py wrapper --duration 30 -o /tmp/profile_wrapper.py

# Parse NCCL flight recorder dumps
python3 profiling/nccl_flight_recorder.py parse /tmp/nccl_trace_rank0.pkl

# Parse torch.compile traces
python3 profiling/tlparse_runner.py parse ./torch_traces/
```

See [profiling/README.md](profiling/README.md) and
[docs/profiling.md](docs/profiling.md) for details.

## Configuration

Everything is configured via environment variables in your cluster's
`env.sh` — nothing is hardcoded to a specific cluster. See
[docs/configuration.md](docs/configuration.md) for the full reference
and [examples/](examples/) for templates.

## Using with Claude Code

This repo includes Claude Code skills for guided assessment:

```
/cluster-assessor my-cluster my-namespace
```

Checks deployment health, runs experiments, analyzes data, and writes
the assessment.

## Repo Layout

```
toolkit/            Diagnostics toolkit (Python 3, stdlib only)
advisor/            Advisory tools: plan, diagnose, rebalance, health
profiling/          PyTorch profiling wrappers
manifests/          K8s manifests for real GPU P/D deployment
manifests/sim/      K8s manifests for inference-sim (no GPU)
examples/           Config templates
clusters/           Per-cluster config and results
scripts/deploy.sh   Deploy manifests with cluster-specific config
docs/               Configuration and profiling guides
```

## See Also

- [llm-d](https://github.com/llm-d/llm-d) — the project
- [llm-d-benchmark](https://github.com/llm-d/llm-d-benchmark) — official
  benchmark framework (Helm-based, CI/CD scale). This toolkit is different:
  lightweight, portable, meant for hands-on cluster validation.
- [llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim) — GPU-free
  vLLM simulator
