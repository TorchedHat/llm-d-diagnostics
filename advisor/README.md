# disagg-advisor

**The only tool that tells you whether to disaggregate.**

Every other tool in the ecosystem benchmarks, simulates, or monitors -- none of them answer the question. disagg-advisor does: give it a model and a GPU, and it tells you yes or no, with the math to back it up.

No dependencies. No setup. One command.

```bash
python3 advisor/plan.py --model meta-llama/Llama-3.1-8B-Instruct --gpu-type h100
```

## What you get

### Capacity planning -- no cluster required

```
============================================================
  CAPACITY PLAN: TinyLlama/TinyLlama-1.1B-Chat-v1.0
============================================================
  Target: 1.0 req/s, TTFT < 500ms
  GPU: T4    Confidence: measured

  Option A: MONOLITHIC
    TP=1, Instances=5
    GPUs: 5    Cost: $2.65/hr

  Option B: DISAGGREGATED (2P + 5D)
    GPUs: 7    Cost: $3.71/hr

  RECOMMENDATION: MONOLITHIC
    - Based on measured data: mono 0.21 req/s, disagg 0.21 req/s
    - Mono uses fewer GPUs with better per-GPU efficiency
============================================================
```

Backed by measured baselines from 8 models across 6 families. Costs across 6 cloud providers with live pricing from cloud-gpus.com. Works for any HuggingFace model -- fetches architecture, computes parallelism, interpolates from the nearest measured data.

```bash
python3 advisor/plan.py --model mistralai/Mixtral-8x7B-v0.1 --gpu-type h100 --throughput 50 --ttft-slo 200
```

### Root-cause diagnosis -- every fix is copy-pasteable

Your disagg deployment broke at 2am. Instead of reading logs for an hour:

```bash
python3 advisor/diagnose.py --namespace prod
```

```
  [CRIT] vLLM image version mismatch across pods
  Evidence: Different images: vllm-prefill: v0.18.1, vllm-decode: v0.17.0
  Cause:    Mismatched vLLM/NIXL versions cause handshake failures (ai-dynamo#6671)
  Fix:      oc set image deployment/vllm-decode vllm=vllm/vllm-openai:v0.31.0 -n prod
            (auto-fixable: copy-paste the command above)

  [CRIT] NIXL transfer failures detected on vllm-decode-0
  Evidence: 12 failed NIXL KV transfers
  Fix:      oc logs vllm-decode-0 -c vllm | grep -i 'nixl|transfer|error' | tail -20
```

Every check maps to a real production bug from llm-d, NVIDIA Dynamo, or vLLM. Every issue comes with the GitHub reference and the exact command to fix it.

### Prefill/decode ratio optimization -- know when to scale

```bash
python3 advisor/rebalance.py --namespace prod
```

```
  Current: 1P + 2D

  decode  vllm-decode-0: KV=85%, waiting=0, running=5
  decode  vllm-decode-1: KV=82%, waiting=1, running=4

  RECOMMENDATION: ADD DECODE REPLICA
    - Decode KV util (84%) is high while prefill queue is low (0).
    - Decode is the bottleneck -- add a decode replica.
```

Watch mode tracks the KV trend over time and tells you *when* you'll need to scale:

```bash
python3 advisor/rebalance.py --namespace prod --watch
```

### Continuous health monitoring

```bash
python3 advisor/health.py --namespace prod --model meta-llama/Llama-3.1-8B-Instruct
```

Combines passive metrics scraping (NIXL failures, KV pressure, queue balance, transfer duration) with active synthetic probing. Runs until you stop it.

## How it compares

| | disagg-advisor | AIConfigurator | llm-d-benchmark | inference-perf | GuideLLM |
|---|---|---|---|---|---|
| Gives a verdict | **Yes** | No | No | No | No |
| Uses real hardware | **Yes** | No (simulation) | Yes | Yes | Yes |
| Cost estimation | **6 providers + live** | No | No | No | No |
| Diagnoses failures | **Yes** | No | No | No | No |
| Prefill/decode rebalancing | **Yes** | No | No | No | No |
| Dependencies | **None (stdlib)** | pip + GPU | Helm | Go + Helm | pip |

## All commands

```bash
# No cluster needed
python3 advisor/plan.py --model MODEL --gpu-type GPU [--throughput N] [--ttft-slo MS] [--provider CLOUD]

# Needs a running disagg cluster
python3 advisor/diagnose.py --namespace NS [--model MODEL]
python3 advisor/rebalance.py --namespace NS [--watch] [--interval S] [--duration S]
python3 advisor/health.py --namespace NS --model MODEL [--interval S] [--duration S]

# Or via the toolkit dispatcher
./toolkit/run.sh clusters/my-cluster plan --model MODEL --gpu-type GPU
./toolkit/run.sh clusters/my-cluster diagnose
./toolkit/run.sh clusters/my-cluster rebalance
./toolkit/run.sh clusters/my-cluster health
```

## Roadmap

- [ ] RDMA / InfiniBand measured baselines
- [ ] H100 / H200 / B200 measured baselines
- [ ] Large models (70B+, Mixtral 8x22B, DeepSeek-V3)
- [ ] Multi-node TP/PP (asymmetric parallelism)
- [ ] Workload profiling from production traffic logs
- [ ] Quantization-aware planning (INT4/FP8)
- [ ] Upstream into llm-d ([#48](https://github.com/llm-d/llm-d/issues/48), [#611](https://github.com/llm-d/llm-d/issues/611))
- [ ] Helm / KEDA integration for automated P/D scaling

## Tests

```bash
python3 -m pytest advisor/tests/ -v
```

Covers every decision path: capacity planning (measured + extrapolated), all diagnostic checks, all rebalance thresholds, probe baseline tracking, pricing (static + live fetch), and the cluster adapter.

## Attribution

Advisory tools originated in [disagg-advisor](https://github.com/llm-d/disagg-advisor) by Arkadip Maitra.
