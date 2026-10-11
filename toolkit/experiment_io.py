"""Standard-library measurement helpers for explicit experiment configurations."""

import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import pathlib
import random
import re
import secrets
import subprocess
import time
import urllib.error
import urllib.request
from itertools import pairwise

WORDS = "apple river mountain cloud forest stone water light garden orange music table chair train road summer winter ocean moon earth bird window paper quiet green red blue morning evening story city village book grass".split()
SAMPLE = re.compile(r"^([^\s{]+)(\{.*\})?\s+(\S+)(?:\s+.*)?$")


def positive(value, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number")
    if not math.isfinite(value) or value <= 0 or (integer and int(value) != value):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")
    return int(value) if integer else value


def url(value, name):
    from urllib.parse import urlparse

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{name} must be an HTTP(S) URL")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError(f"{name}: use EXPERIMENT_API_KEY for authentication")
    return value


def http(endpoint, payload=None, headers=None, timeout=300):
    extra = {"Content-Type": "application/json", **(headers or {})}
    token = os.environ.get("EXPERIMENT_API_KEY")
    if token:
        extra.setdefault("Authorization", f"Bearer {token}")
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=extra,
    )
    return urllib.request.urlopen(request, timeout=timeout)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_config(path):
    config = json.loads(pathlib.Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("configuration must be an object")
    config.setdefault("nonce", secrets.token_hex(16))
    if not isinstance(config["nonce"], str) or not config["nonce"]:
        raise ValueError("nonce must be a nonempty string")
    if not isinstance(config.get("model"), str) or not config["model"]:
        raise ValueError("model is required")
    url(config.get("tokenize_url", ""), "tokenize_url")
    config.setdefault("timeout", 300)
    positive(config["timeout"], "timeout")
    config.setdefault("warmup", 3)
    if (
        isinstance(config["warmup"], bool)
        or not isinstance(config["warmup"], int)
        or config["warmup"] < 1
    ):
        raise ValueError("warmup must be at least 1")
    for target in _targets(config):
        headers = target.get("headers", {})
        if not isinstance(headers, dict) or any(not isinstance(v, str) for v in headers.values()):
            raise ValueError("headers must map names to strings")
        if any(k.lower() in {"authorization", "cookie", "proxy-authorization"} for k in headers):
            raise ValueError(
                "use EXPERIMENT_API_KEY; credentials must not be saved in configurations"
            )
    return config


def _targets(config):
    if "arms" in config:
        if not isinstance(config["arms"], list) or any(
            not isinstance(a, dict) for a in config["arms"]
        ):
            raise ValueError("arms must be a list of objects")
        return [t for arm in config["arms"] for t in arm.get("targets", [])]
    return [config[k] for k in ("prefill", "decode") if k in config]


def initialize(directory, config, script):
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config)
    here = pathlib.Path(__file__).resolve().parent
    commit = os.environ.get("TOOLKIT_COMMIT", "unknown")
    if commit == "unknown":
        try:
            result = subprocess.run(
                ["git", "-C", str(here), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
                dirty = subprocess.run(
                    ["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if dirty.stdout.strip():
                    commit += "-dirty"
        except OSError:
            pass
    write_json(
        directory / "run-info.json",
        {
            "toolkit_commit": commit,
            "config_sha256": digest(config),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (here / pathlib.Path(script).name, pathlib.Path(__file__))
            },
        },
    )
    return directory


def manifest(directory):
    directory = pathlib.Path(directory)
    records = {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p != directory / "manifest.json"
    }
    write_json(directory / "manifest.json", records)


def verify_manifest(directory):
    directory = pathlib.Path(directory)
    records = json.loads((directory / "manifest.json").read_text())
    actual = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p != directory / "manifest.json"
    }
    if set(records) != actual:
        raise ValueError("manifest file list differs from the artifact directory")
    for name, expected in records.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"missing or invalid manifest path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"artifact checksum mismatch: {path}")


def prepare(config, tokens, key):
    seed = digest((config["nonce"], config["model"], tokens, key))
    rng = random.Random(int(seed, 16))
    text = seed + " " + " ".join(rng.choice(WORDS) for _ in range(tokens * 2))
    with http(
        config["tokenize_url"],
        {"model": config["model"], "prompt": text},
        timeout=config["timeout"],
    ) as response:
        ids = json.load(response)["tokens"][:tokens]
    if len(ids) != tokens or any(
        isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids
    ):
        raise ValueError("tokenizer did not return the requested number of token IDs")
    return ids


def send(config, target, prompt, index, salt, output_tokens, extra=None):
    payload = {
        "model": config["model"],
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "cache_salt": digest((config["nonce"], salt)),
        **(extra or {}),
    }
    start = time.monotonic_ns()
    started_unix_ns = time.time_ns()
    chunks, ids = [], []
    status, error, usage, request_id = 0, "", None, None
    complete = False
    try:
        with http(target["url"], payload, target.get("headers"), config["timeout"]) as response:
            status = response.status
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    complete = True
                    break
                item = json.loads(line[6:])
                if item.get("error"):
                    raise RuntimeError(str(item["error"]))
                if item.get("id"):
                    if request_id is not None and item["id"] != request_id:
                        raise RuntimeError("response request ID changed within the stream")
                    request_id = item["id"]
                if item.get("usage"):
                    usage = item["usage"]
                choices = item.get("choices", [])
                if len(choices) > 1:
                    raise RuntimeError("expected one completion choice")
                for choice in choices:
                    part = choice.get("token_ids") or []
                    if not isinstance(part, list) or any(type(i) is not int or i < 0 for i in part):
                        raise RuntimeError("invalid completion token IDs")
                    if choice.get("text") and not part:
                        raise RuntimeError(
                            "token IDs missing; streamed chunks cannot be counted as tokens"
                        )
                    if part:
                        chunks.append(
                            {"seconds": (time.monotonic_ns() - start) / 1e9, "tokens": len(part)}
                        )
                        ids.extend(part)
        if not complete or not usage:
            raise RuntimeError("missing stream completion or usage")
        if usage.get("prompt_tokens") != len(prompt):
            raise RuntimeError(f"unexpected prompt usage: {usage}")
        if usage.get("completion_tokens") != output_tokens or len(ids) != output_tokens:
            raise RuntimeError(f"incomplete output or token-ID count: {usage}")
    except urllib.error.HTTPError as exc:
        status, error = exc.code, exc.read().decode(errors="replace")[:1000]
    except Exception as exc:
        error = str(exc)
    wall_ms = (time.monotonic_ns() - start) / 1e6
    ttft = chunks[0]["seconds"] * 1000 if chunks else None
    itl = (
        (chunks[-1]["seconds"] - chunks[0]["seconds"]) * 1000 / (len(ids) - 1)
        if len(ids) > 1 and chunks and chunks[0]["tokens"] == 1
        else None
    )
    single = len(chunks) == len(ids) and all(c["tokens"] == 1 for c in chunks)
    gaps = [(b["seconds"] - a["seconds"]) * 1000 for a, b in pairwise(chunks)]
    return {
        "request": index,
        "target": target.get("label", "target"),
        "started_unix_ns": started_unix_ns,
        "started_monotonic_ns": start,
        "finished_monotonic_ns": start + int(wall_ms * 1e6),
        "status": status,
        "success": status == 200 and not error,
        "error": error,
        "usage": usage,
        "server_request_id": request_id,
        "prompt_sha256": digest(prompt),
        "completion_tokens": len(ids),
        "ttft_ms": ttft,
        "itl_mean_ms": itl,
        "total_ms": wall_ms,
        "itl_p99_ms": percentile(gaps, 0.99) if single else None,
        "single_token_chunks": single,
        "stream_chunks": chunks,
        "generated_token_ids": ids,
    }


def percentile(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def snapshot(endpoints):
    data = {"time_ns": time.time_ns(), "pods": {}}
    for name, endpoint in endpoints.items():
        with http(endpoint, timeout=10) as response:
            text = response.read().decode()
        data["pods"][name] = [
            line
            for line in text.splitlines()
            if line.startswith(("vllm:", "process_start_time_seconds"))
        ]
    return data


def counter(data, pod, metric, source=None):
    values, present = [], False
    for line in data["pods"][pod]:
        match = SAMPLE.match(line)
        if match and match[1] == metric:
            present = True
            if source is None or f'source="{source}"' in (match[2] or ""):
                value = float(match[3])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid counter {pod}/{metric}")
                values.append(value)
    if not present:
        raise ValueError(f"missing metric {pod}/{metric}")
    return sum(values)


def delta(before, after, pod, metric, source=None):
    if set(before["pods"]) != set(after["pods"]):
        raise ValueError("metrics endpoints changed")
    for name in before["pods"]:
        starts = [
            [line for line in s["pods"][name] if line.startswith("process_start_time_seconds")]
            for s in (before, after)
        ]
        if starts[0] != starts[1]:
            raise ValueError(f"process restarted: {name}")
    value = counter(after, pod, metric, source) - counter(before, pod, metric, source)
    if value < -1e-9:
        raise ValueError(f"counter reset: {pod}/{metric}")
    return max(0, value)


def closed_loop(concurrency, prompts, targets, fn, directory):
    results, active = [], {}
    cursor = 0
    with (
        gzip.open(pathlib.Path(directory) / "requests.jsonl.gz", "wt") as file,
        concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool,
    ):
        started = time.monotonic()
        while cursor < min(concurrency, len(prompts)):
            active[pool.submit(fn, targets[cursor % len(targets)], prompts[cursor], cursor)] = (
                cursor
            )
            cursor += 1
        while active:
            completed, _ = concurrent.futures.wait(
                active, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in completed:
                active.pop(future)
                row = future.result()
                results.append(row)
                file.write(json.dumps(row, allow_nan=False) + "\n")
                file.flush()
                if cursor < len(prompts):
                    active[
                        pool.submit(fn, targets[cursor % len(targets)], prompts[cursor], cursor)
                    ] = cursor
                    cursor += 1
        wall_seconds = time.monotonic() - started
    return results, wall_seconds


def save_prompts(directory, prompts):
    with gzip.open(pathlib.Path(directory) / "prompts.jsonl.gz", "wt") as file:
        for index, prompt in enumerate(prompts):
            file.write(
                json.dumps({"request": index, "token_ids": prompt, "sha256": digest(prompt)}) + "\n"
            )
