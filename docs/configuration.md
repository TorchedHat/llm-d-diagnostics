# Configuration

Everything is configured via environment variables so nothing is hardcoded
to a specific cluster.

## Global (all experiments)

| Variable | Default | What it does |
|----------|---------|-------------|
| `NS` | `default` | Kubernetes namespace |
| `MODEL` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Model name for API requests |
| `SIM` | (unset) | Set to `1` for inference-sim mode (HTTP, no TLS) |
| `BASELINE_URL` | `http://vllm-prefill-svc:8000/v1/completions` (`SIM=1`: port 8100) | Prefill direct endpoint |
| `DISAGG_D1_URL` | `http://vllm-decode-svc:8000/v1/completions` | Decode-1 through sidecar (scheme follows `SIDECAR_SCHEME`) |
| `DISAGG_D2_URL` | `http://vllm-decode-2-svc:8000/v1/completions` | Decode-2 through sidecar (scheme follows `SIDECAR_SCHEME`) |
| `EPP_URL` | (unset) | Full OpenAI completions URL exposed by the Gateway HTTPRoute; enables EPP comparison arms |
| `ROUTING_MODE` | `manual` | `manual` or `epp`; selects the route for exp4 and `send_disagg()`. Comparison experiments always retain their manual controls. |
| `EPP_GATEWAY_NAME` | (required to install EPP) | Existing InferencePool-capable Gateway to attach the HTTPRoute to |
| `EPP_GATEWAY_NAMESPACE` | `$NS` | Namespace containing that Gateway |
| `EPP_ROUTER_VERSION` | `v0.11.0` | llm-d-router chart and image version installed by `deploy-epp.sh` |
| `EPP_RELEASE_NAME` | `llm-d-epp` | Helm release name |
| `EPP_SERVICE_NAME` | `llm-d-epp-epp` | In-cluster EPP Service name derived from `EPP_RELEASE_NAME` |
| `EPP_METRICS_URL` | `http://llm-d-epp-epp:9090` | EPP Prometheus endpoint scraped when `EPP_URL` is set |
| `RUNS` | `20` | Measured runs per config |
| `WARMUP` | `3` | Warmup requests (discarded) |
| `MAX_TOKENS` | `20` | Max completion tokens |
| `DATA_DIR` | `data` | Where CSV results get written |

## Per-experiment overrides

| Variable | Default | Experiment | What it does |
|----------|---------|------------|-------------|
| `PROMPT_LENGTHS` | `10,50,100,500,1000` | exp1 | Comma-separated prompt lengths |
| `CONCURRENCY_LEVELS` | `1,2,4,8,16` | exp2 | Comma-separated concurrency levels |
| `HEAVY_PROMPT_TOKENS` | `1000` | exp3 | Heavy request prompt size |
| `LIGHT_PROMPT_TOKENS` | `10` | exp3 | Light request prompt size |
| `SWEEP_LENGTHS` | `10,50,100,250,500,1000,2000,4096` | exp5 | Sequence lengths to sweep |
| `QPS_LEVELS` | `1,2,4,8,12,16,24,32` | exp6 | QPS levels to test |
| `DURATION_S` | `30` (exp6), `60` (exp7) | exp6, exp7 | Seconds per config/QPS level |
| `SLO_MULT` | `2.0` | exp6 | Saturation threshold (p99 > N × baseline p50) |
| `QPS` | `4` | exp7 | Target arrival rate |
| `LONG_PCT` | `20` | exp7 | Percentage of long prompts in mix |
| `SHORT_TOKENS` | `10` | exp7 | Short prompt token count |
| `LONG_TOKENS` | `500` | exp7 | Long prompt token count |
| `TEST_CLIENT` | `test-client` | exp4 | Pod name for test client |
| `PREFILL_DEPLOY` | `vllm-prefill` | exp4 | Prefill deployment name |
| `DECODE1_DEPLOY` | `vllm-decode` | exp4 | Decode-1 deployment name |
| `DECODE2_DEPLOY` | `vllm-decode-2` | exp4 | Decode-2 deployment name |
| `LOAD_REQUESTS` | `8` | exp4e | Concurrent requests for failure-under-load |
| `NETEM_DELAY_MS` | `100` | exp4g | Added latency for slow-network test |
| `NETEM_LOSS_PCT` | `10` | exp4g | Packet loss percentage |
| `KEEPALIVE_RUNS` | `3` | exp4h | UCX keepalive measurement runs |
| `KILL_DELAYS_MS` | `100,200,500` | exp4i | Kill delays for mid-transfer test |
| `LOAD_QPS` | `4` | exp4k, exp4l | Target QPS for sustained load tests |
| `LOAD_DURATION` | `60` | exp4k | Duration of load test in seconds |
| `KILL_AT_S` | `10` | exp4k | Seconds before prefill kill |
| `ROLLOUT_DURATION` | `90` | exp4l | Duration of rolling update test |
| `CACHE_LENGTHS` | `100,500,1000` | exp8 | Prompt lengths for prefix cache test |
| `CONV_RUNS` | `5` | exp8 | Multi-turn conversation runs |
| `CACHE_DECAY_S` | `30` | exp8 | Delay between cache hit checks |
| `LOAD_RUNS` | `3` | exp9 | Number of cold-start measurements |
| `DECODE_SELECTOR` | `app=vllm-decode` | exp9 | Pod selector for decode pod to kill |
| `STARTUP_TIMEOUT` | `600` | exp9 | Max wait for pod restart (seconds) |
| `EVICTION_DELAYS` | `10,30,60` | exp10 | Delay sweep for cache eviction |
| `EVICTION_RUNS` | `5` | exp10 | Runs per delay interval |
| `CACHE_PROMPT_TOKENS` | `500` | exp10 | Prompt length for eviction test |
| `BG_LOAD` | `0` | exp10 | Background QPS during eviction test |
| `PRESSURE_PROMPTS_N` | `20` | exp10 | Eviction pressure prompts |

Full list in [`toolkit/client.py`](../toolkit/client.py) and each experiment script.

## Model configuration (Kubernetes manifests)

The model name, context length, GPU memory utilization, and dtype are
defined once in a ConfigMap (`manifests/00-model-config.yaml`). All
deployments reference it. To switch models, edit the ConfigMap and
re-apply:

```yaml
# manifests/00-model-config.yaml
data:
  MODEL_NAME: "meta-llama/Llama-3.1-8B-Instruct"
  MAX_MODEL_LEN: "4096"
  GPU_MEMORY_UTILIZATION: "0.9"
  DTYPE: "float16"
```

When deploying with `deploy.sh`, the ConfigMap is generated from `env.sh`
values — you don't need to edit the yaml. If deploying manually with
`oc apply -f manifests/`, edit `00-model-config.yaml` directly and ensure
`MODEL` in `env.sh` matches `MODEL_NAME` in the ConfigMap.

## Deployment configuration

| Variable | Default | What it does |
|----------|---------|-------------|
| `VLLM_IMAGE` | `vllm/vllm-openai:v0.31.0` | Latest upstream vLLM release, CUDA 13.0 build; override for your accelerator. llm-d v0.10.0's release matrix lists v0.30.0. |
| `SIDECAR_IMAGE` | `ghcr.io/llm-d/llm-d-router-disagg-sidecar:v0.11.0` | llm-d Router disaggregation sidecar image |
| `GPU_ALLOCATION_MODE` | `classic` | GPU request mechanism: `dra` or `classic` |
| `GPU_COUNT` | `1` | GPUs requested per vLLM pod; used for vLLM tensor parallelism |
| `GPU_RESOURCE_NAME` | `nvidia.com/gpu` | Classic device-plugin resource name; ignored in DRA mode |
| `SIDECAR_SCHEME` | `http` | Sidecar listener and toolkit URL scheme (`http` or `https`) |
| `MODEL_CACHE_SIZE` | `50Gi` | PVC size for model weights cache |
| `STORAGE_CLASS` | (cluster default) | Kubernetes StorageClass for model-cache PVC |
| `MAX_MODEL_LEN` | `2048` | Maximum sequence length |
| `GPU_MEMORY_UTILIZATION` | `0.8` | vLLM GPU memory fraction |
| `DTYPE` | `float16` | Model weight precision |

`STORAGE_CLASS` must support `ReadWriteMany` if pods are spread across
nodes (anti-affinity). On OpenShift OCS, use `ocs-storagecluster-cephfs`.
If unset, the cluster's default StorageClass is used.

In DRA mode, each vLLM pod references the namespaced `ResourceClaimTemplate`
named `gpu-${GPU_COUNT}` and attaches the resulting claim to its vLLM
container. The template must exist before deployment. For example,
`GPU_COUNT=2` uses `resourceClaimTemplateName: gpu-2` and sets vLLM tensor
parallelism to 2. Classic mode requests `GPU_COUNT` units of
`GPU_RESOURCE_NAME` instead.

## Cluster config file

Each cluster has an `env.sh` that sets the variables for that deployment:

```bash
# clusters/my-cluster/env.sh
export NS=my-namespace
export MODEL="my-org/my-model"
export DATA_DIR="clusters/my-cluster/data"
export STORAGE_CLASS="my-rwx-storage-class"  # must support ReadWriteMany
export PREFILL_HOST="vllm-prefill-svc.${NS}.svc.cluster.local:8000"
```

Source it before running experiments directly, or let `run.sh` source it
automatically.

## EPP integration

The EPP chart attaches an `HTTPRoute` to a Gateway that the cluster operator
has already installed and configured for `InferencePool` backends. Set
`EPP_GATEWAY_NAME` and `EPP_GATEWAY_NAMESPACE`, then install and remove the
namespaced router release with:

```bash
./scripts/deploy-epp.sh clusters/my-cluster
./scripts/undeploy-epp.sh clusters/my-cluster
```

The installer checks Gateway readiness, model pod labels, namespace permissions,
and a server dry-run of the rendered resources before installation. The dry-run
also checks RBAC escalation and binding restrictions. It pins llm-d-router/EPP to `v0.11.0` by default,
matching the current disaggregation sidecar. The EPP pool selects only model
pods labeled `llm-d-diagnostics.ai/model-server=true`, uses pod port 8000 for
both roles, and requests P/D disaggregation with the always-disagg decider. The
prefill Service is headless, so clients resolve pod IPs and connect directly to
port 8000. Inference-sim uses port 8100.

After installation, take the Gateway address from its status and set
`EPP_URL` to that address plus `/v1/completions`. With `EPP_URL` configured,
the latency, decomposition, throughput, isolation, saturation, mixed-workload,
and sequence-sweep experiments add a separate `DISAGG-EPP`/`E-EPP-gateway`
arm alongside their manual-routing controls. `exp8` adds a cache-locality phase
through the Gateway, and `exp14` measures manual-versus-EPP routing under
concurrency. `exp4` uses manual headers by default; set `ROUTING_MODE=epp` to
send its in-cluster fault probes through the Gateway. The metrics collector
saves raw EPP snapshots and samples the
`llm_d_epp_scheduler_e2e_duration_seconds` histogram sum/count; it does not
estimate histogram percentiles by subtracting summary quantiles.

The upstream handler can serve on decode alone when no prefill endpoint is
available. Check NIXL transfer counters during fault tests to identify this
fallback; an HTTP 200 response alone does not establish that KV transfer occurred.

The client-vs-manual latency difference includes Gateway proxying and EPP
scheduling. The EPP histogram measures scheduler latency only. Keep both when
reporting router overhead.

## Security notes

This toolkit is designed for **cluster diagnostics by trusted
operators**. A few things to be aware of:

- The deployment defaults to HTTP inside the cluster. Set `SIDECAR_SCHEME=https`
  to enable the sidecar's self-signed TLS listener; toolkit clients disable
  certificate verification for that endpoint (`ssl.CERT_NONE`).
- **Containers run as root** because vLLM requires it for CUDA and NIXL
  memory registration. OpenShift SCCs enforce additional restrictions at
  the cluster level.
- **Model cache PVC is read-write** because vLLM downloads model weights
  on first startup. A production deployment should pre-populate the cache
  and mount read-only.
