#!/usr/bin/env python3
"""Run a finite, paired streaming comparison from a cluster client pod.

This worker emits JSONL incrementally; it never retries an inference request.
"""

import concurrent.futures
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request


def emit(value):
    sys.stdout.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def post(url, payload, headers=None, timeout=180):
    request = urllib.request.Request(
        url, json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})}, method="POST")
    return urllib.request.urlopen(request, timeout=timeout)


def one(config, arm, concurrency, block, number, prompt, warmup=False):
    url = config["mono_urls"][number % len(config["mono_urls"])] if arm == "MONO" else config["pd_url"]
    route = config.get("pd_route", "manual")
    if route not in {"manual", "epp"}:
        raise ValueError("unsupported P/D route")
    headers = {"x-prefiller-host-port": config["prefill_host"]} if arm == "PD" and route == "manual" else {}
    payload = {"model": config["model"], "prompt": prompt, "max_tokens": config["output_tokens"],
               "temperature": 0, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}, "return_token_ids": True}
    row = {"event": "request", "arm": arm, "concurrency": concurrency, "block": block,
           "number": number, "warmup": warmup,
           "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
           "url": url, "ok": False, "status": None, "done": False, "ttft_ms": None,
           "completion_tokens": None, "prompt_tokens": None, "elapsed_ms": None, "error": None,
           "tpot_ms": None, "timed_token_count": 0, "coalesced_chunks": 0}
    start = time.monotonic()
    first = None
    last = None
    first_chunk_tokens = None
    token_ids_complete = True
    try:
        with post(url, payload, headers, timeout=config["timeout_seconds"]) as response:
            row["status"] = response.status
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    row["done"] = True
                    break
                chunk = json.loads(data)
                for choice in chunk.get("choices", []):
                    ids = choice.get("token_ids") or []
                    if ids or choice.get("text"):
                        now = time.monotonic()
                        if first is None:
                            first = now
                            first_chunk_tokens = len(ids)
                        last = now
                        if ids:
                            row["timed_token_count"] += len(ids)
                            row["coalesced_chunks"] += int(len(ids) > 1)
                        else:
                            token_ids_complete = False
                usage = chunk.get("usage")
                if usage:
                    row["completion_tokens"] = usage.get("completion_tokens")
                    row["prompt_tokens"] = usage.get("prompt_tokens")
        row["elapsed_ms"] = (time.monotonic() - start) * 1000
        row["ttft_ms"] = (first - start) * 1000 if first is not None else None
        tokens = row["completion_tokens"]
        if (token_ids_complete and first_chunk_tokens == 1 and first is not None and last is not None
                and isinstance(tokens, int) and tokens > 1 and row["timed_token_count"] == tokens):
            row["tpot_ms"] = (last - first) * 1000 / (tokens - 1)
        row["ok"] = bool(row["status"] == 200 and row["done"] and first is not None
                         and row["completion_tokens"] == config["output_tokens"]
                         and row["prompt_tokens"] == len(prompt))
        if not row["ok"]:
            row["error"] = "incomplete stream or unexpected token counts"
    except urllib.error.HTTPError as exc:
        row["status"] = exc.code
        row["elapsed_ms"] = (time.monotonic() - start) * 1000
        row["error"] = str(exc)[:500]
    except (OSError, ValueError, urllib.error.URLError) as exc:
        row["elapsed_ms"] = (time.monotonic() - start) * 1000
        row["error"] = str(exc)[:500]
    emit(row)
    return row


def make_prompts(base, length, count, seed, scope):
    """Generate distinct exact-length synthetic token sequences for one block."""
    if not base or length < 1 or count < 1:
        raise ValueError("empty tokenizer output or invalid prompt size")
    prompts = []
    for number in range(count):
        key = json.dumps([seed, scope, number], separators=(",", ":"))
        rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))
        prompts.append(rng.choices(base, k=length))
    return prompts


def main():
    config = json.loads(sys.argv[1])
    # No environment or credentials are copied into the evidence.
    emit({"event": "start", "config": config})
    text = " ".join(f"calibration seed {config['seed']} sample {i} compute data network serving"
                    for i in range(3000))
    tokenize_url = config["mono_urls"][0].rsplit("/v1/completions", 1)[0] + "/tokenize"
    with post(tokenize_url, {"model": config["model"], "prompt": text}) as response:
        base = json.load(response)["tokens"]
    n = config["input_tokens"]
    for arm in ("MONO", "PD"):
        prompts = make_prompts(base, n, 2, config["seed"], f"warmup-{arm}")
        for i in range(2):
            if not one(config, arm, 1, -1, i, prompts[i], warmup=True)["ok"]:
                raise RuntimeError(f"{arm} warmup failed")
    rng = random.Random(config["seed"])
    for concurrency in config["concurrencies"]:
        for block in range(config["blocks"]):
            prompts = make_prompts(base, n, config["requests"], config["seed"],
                                   f"c{concurrency}-block{block}")
            emit({"event": "inputs", "concurrency": concurrency, "block": block, "tokens": prompts})
            arms = ["MONO", "PD"]
            rng.shuffle(arms)
            for arm in arms:
                started = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    jobs = [pool.submit(one, config, arm, concurrency, block, i, prompt)
                            for i, prompt in enumerate(prompts)]
                    rows = [job.result() for job in jobs]
                emit({"event": "cell", "arm": arm, "concurrency": concurrency, "block": block,
                      "requests": len(rows), "success": sum(r["ok"] for r in rows),
                      "wall_seconds": time.monotonic() - started})
    emit({"event": "complete"})


if __name__ == "__main__":
    main()
