# Paired serving profiles

Compare existing two-GPU monolithic and prefill/decode deployments using paired,
synthetic token sequences. The controller uses two distinct one-GPU monolithic
pods and one one-GPU pod for each P/D role. All four serving pods remain resident;
this is two active GPUs per arm and four reserved GPUs in the experiment.

The controller needs local Python, `oc` or `kubectl`, pod get/exec access, and an
existing client pod with Python and HTTP access. Serving pods need Python and
`nvidia-smi`. They must serve the same model and tokenizer, weights, quantization,
TP1 and relevant serving and cache settings. The operator must check those
settings and allocation/isolation independently. Four distinct visible GPU IDs
are checked, but they do not establish exclusive allocation.

```sh
python3 toolkit/serving_profile.py --namespace "$NS" --client-pod "$CLIENT_POD" \
  --client-container "$CLIENT_CONTAINER" --serving-container vllm \
  --mono-pods "$MONO1,$MONO2" --prefill-pod "$PREFILL_POD" \
  --decode-pod "$DECODE_POD" --model "$MODEL" --output-dir /tmp/serving-profile \
  --input-tokens 4096 --output-tokens 64 --concurrencies 2,8 --requests 32 --blocks 3
```

The P/D arm defaults to an explicit prefill header sent to the decode sidecar. For
an existing EPP endpoint, add `--pd-route epp --epp-url
http://router:8080/v1/completions --router-pod "$ROUTER_POD"`. The worker sends
no manual prefill header in EPP mode. The controller tracks the specified router's
UID, image IDs and restarts; it does not prove the URL or Service reaches only that
pod or that EPP selected the declared backends. Verify router membership, routing
decisions and application KV counters independently. The tool does not create or
configure EPP, Gateway, InferencePool, networking or serving pods.

Each concurrency/block uses matched exact-token prompts, randomizes arm order,
and runs finite closed-loop traffic with two warmups per arm. Per-request output
contains first observed output time, request-mean time per output token when
available, status, completion and token usage. Chunk coalescing prevents observing
individual token gaps; a first chunk containing several tokens makes request-mean
time per output token unavailable. Inference requests are never automatically
retried. The controller checks cell coverage, paired prompt hashes and successful
request counts before writing `result.json`; failures remain in `raw.jsonl` and a
failed controller writes `error.json`. An output directory must be new.

This is a finite sample, not a production capacity or population SLO estimate. It
also does not establish task quality or validate an advisor prediction. Profiles
include pod and cluster metadata plus synthetic prompt token IDs; review artifacts
before sharing.
