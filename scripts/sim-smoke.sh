#!/usr/bin/env bash
# End-to-end smoke test of the streaming experiments (exp12, exp16) and the
# one-GPU calibration (exp18) against
# llm-d-inference-sim, with no GPU and no cluster.
#
# Starts two simulator processes with known latencies, runs each experiment
# with small parameters, runs analyze.py, and checks that the measured TTFT,
# ITL and output lengths match what the simulator was told to produce.
#
# The DISAGG configs here go straight to a second simulator (no routing
# sidecar), so this checks the toolkit's measurement and CSV plumbing, not
# the P/D path.
#
# Usage:
#   SIM_BIN=/path/to/llm-d-inference-sim ./scripts/sim-smoke.sh
#
# Build the simulator (pure Go, no container needed), matching the version in
# manifests/sim/:
#   git clone --depth 1 --branch v0.11.2 https://github.com/llm-d/llm-d-inference-sim
#   cd llm-d-inference-sim && CGO_ENABLED=0 go build -o llm-d-inference-sim \
#       cmd/llm-d-inference-sim/main.go
set -euo pipefail

SIM_BIN="${SIM_BIN:?set SIM_BIN to the llm-d-inference-sim binary}"
PREFILL_PORT="${PREFILL_PORT:-18100}"
DECODE_PORT="${DECODE_PORT:-18001}"
CALIBRATE_PORT="${CALIBRATE_PORT:-18300}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
cleanup() {
    status=$?
    kill $(jobs -p) 2>/dev/null || true
    if [ "$status" -ne 0 ]; then
        for log in "$WORK"/*.log; do
            [ -f "$log" ] && { echo "--- $log"; tail -n 50 "$log"; }
        done
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

# Known latencies: TTFT = 20 ms + 0.1 ms per prompt token, ITL = 10 ms.
SIM_ARGS=(--model=sim-model --max-model-len=4096 --max-num-seqs=8
          --latency-calculator=per-token --prefill-overhead=20ms
          --prefill-time-per-token=100us --inter-token-latency=10ms)

GOMAXPROCS=2 "$SIM_BIN" --port="$PREFILL_PORT" "${SIM_ARGS[@]}" >"$WORK/prefill.log" 2>&1 &
GOMAXPROCS=2 "$SIM_BIN" --port="$DECODE_PORT" "${SIM_ARGS[@]}" >"$WORK/decode.log" 2>&1 &
# Calibration target: step time rises linearly from 10 ms (1 request) to 20 ms
# (8 requests, the simulator's load factor); KV cache 1024 blocks x 16 tokens.
GOMAXPROCS=2 "$SIM_BIN" --port="$CALIBRATE_PORT" "${SIM_ARGS[@]}" --max-model-len=8192 \
    --time-factor-under-load=2 --kv-cache-size=1024 --block-size=16 \
    >"$WORK/calibrate-sim.log" 2>&1 &

for port in "$PREFILL_PORT" "$DECODE_PORT" "$CALIBRATE_PORT"; do
    for _ in $(seq 50); do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
        sleep 0.2
    done
    curl -sf "http://127.0.0.1:$port/health" >/dev/null || {
        echo "simulator on port $port did not start"; cat "$WORK"/*.log; exit 1; }
done

export SIM=1 MODEL=sim-model DATA_DIR="$WORK/data" WARMUP=1 \
    BASELINE_URL="http://127.0.0.1:$PREFILL_PORT/v1/completions" \
    DISAGG_URL="http://127.0.0.1:$DECODE_PORT/v1/completions" \
    PREFILL_HOST="127.0.0.1:$PREFILL_PORT"
mkdir -p "$DATA_DIR"
cd "$REPO_ROOT"

PROMPT_TOKENS=1000 OUTPUT_LENGTHS=10,40 CONCURRENCY=1,4 TOTAL_REQUESTS=4 LOAD_MODE=batch \
    CONFIGS=BASELINE,DISAGG-2D python3 toolkit/exp12_tput_outlen.py >"$WORK/exp12.log" 2>&1
PROMPT_TOKENS=1000 MAX_TOKENS=20 CONCURRENCY_LEVELS=1,4 TOTAL_REQUESTS=4 LOAD_MODE=steady \
    python3 toolkit/exp16_per_token_trace.py >"$WORK/exp16.log" 2>&1
CALIBRATE_URL="http://127.0.0.1:$CALIBRATE_PORT/v1/completions" \
    PREFILL_LENGTHS=128,256,512,1024,2048 PREFILL_RUNS=3 DECODE_BATCHES=1,2,4,8 \
    DECODE_PROMPT_TOKENS=128,512 DECODE_OUTPUT_TOKENS=64 DECODE_RUNS=2 \
    OVERLAP_DECODERS=4 OVERLAP_PROMPT_TOKENS=1024 OVERLAP_OUTPUT_TOKENS=128 OVERLAP_RUNS=2 \
    python3 toolkit/exp18_calibrate.py >"$WORK/exp18.log" 2>&1
python3 toolkit/analyze.py "$DATA_DIR" >"$WORK/analyze.log" 2>&1

python3 - "$DATA_DIR" <<'EOF'
import csv, statistics, sys
d = sys.argv[1]

def rows(name):
    return list(csv.DictReader(open(f"{d}/{name}-results.csv")))

problems = []

e12 = rows("exp12")
if any(r["status_code"] != "200" for r in e12):
    problems.append("exp12: failed requests")
if any(r["completion_tokens"] != r["max_tokens"] for r in e12):
    problems.append("exp12: outputs stopped before max_tokens (ignore_eos not honoured?)")
if any(r["server_completion_tokens"] != r["max_tokens"] for r in e12):
    problems.append("exp12: server token count missing or wrong (include_usage not honoured?)")
if {r["run"] for r in e12 if r["concurrency"] == "4"} != {"1", "2", "3", "4"}:
    problems.append("exp12: concurrency 4 did not run one full batch of 4")
c1 = [r for r in e12 if r["concurrency"] == "1"]
ttft = statistics.median(float(r["ttft_ms"]) for r in c1)
itl = statistics.median(float(r["itl_mean_ms"]) for r in c1)
# The simulator's tokenizer may count the prompt differently; allow 70-200 ms.
if not 70 <= ttft <= 200:
    problems.append(f"exp12: TTFT at c=1 is {ttft:.1f} ms, expected ~120 ms")
if not 8 <= itl <= 14:
    problems.append(f"exp12: mean ITL at c=1 is {itl:.1f} ms, expected ~10 ms")

all16 = rows("exp16")
e16 = [r for r in all16 if r["concurrency"] == "1" and int(r["token_idx"]) > 0]
gap = statistics.median(float(r["gap_ms"]) for r in e16)
if not 8 <= gap <= 14:
    problems.append(f"exp16: median gap at c=1 is {gap:.1f} ms, expected ~10 ms")
# Steady load: at c=4, requests after the first wave start with 3 others in flight.
steady = {(r["config"], r["run"]) for r in all16
          if r["concurrency"] == "4" and int(r["run"]) > 4 and r["inflight_at_start"] == "3"}
runs4 = {(r["config"], r["run"]) for r in all16 if r["concurrency"] == "4" and int(r["run"]) > 4}
if not runs4 or len(steady) < 0.8 * len(runs4):
    problems.append(f"exp16: steady load not reached at c=4 ({len(steady)}/{len(runs4)} requests)")


import json
cal = json.load(open(f"{d}/calibration.json"))
steps = {b: s for b, c, s in cal.get("decode_points", []) if c == min(p[1] for p in cal["decode_points"])}
checks = [
    ("prefill fixed cost", cal.get("prefill_t0_s"), 0.010, 0.030),       # configured 20 ms
    ("prefill per token", cal.get("prefill_a_s_per_token"), 8e-5, 1.2e-4),  # configured 100 us
    ("decode step at B=1", steps.get(1), 0.008, 0.013),  # 10 ms
    ("decode step at B=8", steps.get(8), 0.017, 0.024),  # 20 ms
    # The simulator's step time does not depend on context: KV term ~0.
    ("decode KV term", abs(cal.get("decode_kv_s_per_token", 1.0)), 0.0, 2e-6),
    ("KV capacity", cal.get("kv_capacity_tokens"), 16384, 16384),
]
for name, value, lo, hi in checks:
    if value is None or not lo <= value <= hi:
        problems.append(f"exp18: {name} = {value}, expected {lo}..{hi}")
if cal.get("overlap_theta") is None:
    problems.append("exp18: overlap step did not run")

if problems:
    print("SIM SMOKE FAILED:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print(f"SIM SMOKE OK: TTFT {ttft:.1f} ms, ITL {itl:.1f} ms, per-token gap {gap:.1f} ms; "
      f"calibration prefill {cal['prefill_a_s_per_token'] * 1e6:.0f} us/token, "
      f"step B=1..8 {steps[1] * 1000:.1f}..{steps[8] * 1000:.1f} ms, "
      f"KV {cal['kv_capacity_tokens']} tokens")
EOF
