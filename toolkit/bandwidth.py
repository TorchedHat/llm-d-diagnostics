"""Measure prefill cost and request-correlated NIXL payload transfers."""

import argparse
import gzip
import json
import math
import pathlib
import random
import statistics
import time
import urllib.parse

import experiment_io as io

PREFILL = "vllm:request_prefill_time_seconds"
TIME = "vllm:nixl_xfer_time_seconds"
BYTES = "vllm:nixl_bytes_transferred"
POST = "vllm:nixl_post_time_seconds"
SOURCE = "vllm:prompt_tokens_by_source_total"
ERRORS = [
    "vllm:nixl_num_" + name + "_total"
    for name in ("failed_transfers", "failed_notifications", "kv_expired_reqs")
]
SCOPE = "backend payload / request wall time; not NIC wire bandwidth"


def validate(config):
    config["legacy_poster"] = False
    for key, default in (
        ("runs", 30),
        ("aggregate_requests", 100),
        ("tensor_parallel_size", 1),
        ("metrics_timeout", 30),
    ):
        config.setdefault(key, default)
        config[key] = io.positive(config[key], key, integer=key != "metrics_timeout")
    for key, default in (
        ("prompt_lengths", [512, 2000, 8000, 16000]),
        ("concurrency_levels", [1, 4, 16]),
    ):
        config.setdefault(key, default)
        values = config[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty list of distinct integers")
        config[key] = [io.positive(value, key, integer=True) for value in values]
        if len(set(config[key])) != len(config[key]):
            raise ValueError(f"{key} must contain distinct integers")
    if config["aggregate_requests"] < 3 * max(config["concurrency_levels"]):
        raise ValueError("aggregate_requests must be at least 3 * maximum concurrency")
    config.setdefault("aggregate_prompt_tokens", max(config["prompt_lengths"]))
    config["aggregate_prompt_tokens"] = io.positive(
        config["aggregate_prompt_tokens"], "aggregate_prompt_tokens", integer=True
    )
    config.setdefault("prefill_backoff_tokens", 0)
    if type(config["prefill_backoff_tokens"]) is not int or config[
        "prefill_backoff_tokens"
    ] not in (0, 1):
        raise ValueError("prefill_backoff_tokens must be 0 or 1")
    if not isinstance(config.get("transport"), str) or not config["transport"]:
        raise ValueError("transport must describe the configured transport or control")
    endpoints = []
    for role in ("prefill", "decode"):
        target = config.get(role)
        if (
            not isinstance(target, dict)
            or not isinstance(target.get("label"), str)
            or not target["label"]
        ):
            raise ValueError(f"{role} requires label, url, and direct backend metrics_url")
        io.url(target.get("url", ""), role + ".url")
        endpoints.append(io.url(target.get("metrics_url", ""), role + ".metrics_url"))
    if config["prefill"]["label"] == config["decode"]["label"] or endpoints[0] == endpoints[1]:
        raise ValueError(
            "prefill and decode must have distinct labels and direct metrics endpoints"
        )
    if not config["decode"].get("headers", {}).get("x-prefiller-host-port"):
        raise ValueError("decode requires the manual sidecar x-prefiller-host-port header")
    if config.get("rank_trace_url"):
        io.url(config["rank_trace_url"], "rank_trace_url")
        if urllib.parse.urlparse(config["rank_trace_url"]).query:
            raise ValueError("rank_trace_url must not contain query parameters")
    elif config["tensor_parallel_size"] > 1:
        raise ValueError(
            "TP > 1 requires rank_trace_url; rank-duration sums are not request latency"
        )
    if "kv_bytes_per_token" in config:
        config["kv_bytes_per_token"] = io.positive(
            config["kv_bytes_per_token"], "kv_bytes_per_token", integer=True
        )
        config["block_size"] = io.positive(config.get("block_size"), "block_size", integer=True)
        state = config.setdefault("state_bytes_per_request", 0)
        if type(state) is not int or state < 0:
            raise ValueError("state_bytes_per_request must be a nonnegative integer")
    if "reported_port_GB_s" in config:
        io.positive(config["reported_port_GB_s"], "reported_port_GB_s")
    return config


def endpoints(config):
    return {config[role]["label"]: config[role]["metrics_url"] for role in ("prefill", "decode")}


def accounting(config, before, after, count, length, routed):
    p, d = (config[role]["label"] for role in ("prefill", "decode"))

    def change(slot, name, source=None):
        return io.delta(before, after, slot, name, source)

    for slot in (p, d):
        for name in ERRORS:
            if change(slot, name) != 0:
                raise ValueError(f"NIXL error counter increased: {slot}/{name}")
    for slot in (p, d) if routed else (p,):
        if change(slot, SOURCE, "local_cache_hit") != 0:
            raise ValueError(f"prefix cache reuse: {slot}")
    backoff = config.get("prefill_backoff_tokens", 0) if routed else 0
    if change(p, SOURCE, "local_compute") != count * (length - backoff):
        raise ValueError("prefill token count differs; cached or unrelated traffic may be present")
    result = {
        "prefill_ms_sum": change(p, PREFILL + "_sum") * 1000,
        "prefill_observations": change(p, PREFILL + "_count"),
    }
    if result["prefill_observations"] != count or result["prefill_ms_sum"] <= 0:
        raise ValueError("invalid prefill observations")
    if routed:
        result.update(
            nixl_bytes=change(d, BYTES + "_sum"),
            nixl_ms_sum=change(d, TIME + "_sum") * 1000,
            nixl_post_ms_sum=change(d, POST + "_sum") * 1000,
            nixl_transfers=change(d, TIME + "_count"),
        )
        expected = count * config["tensor_parallel_size"]
        if any(change(d, metric + "_count") != expected for metric in (TIME, BYTES, POST)):
            raise ValueError("unexpected transfer observation count")
        if change(d, "vllm:request_success_total") != count:
            raise ValueError("decode completion count differs")
        external = change(d, SOURCE, "external_kv_transfer")
        recomputed = change(d, SOURCE, "local_compute")
        if not count * (length - 1) <= external <= count * length or recomputed > count:
            raise ValueError("decode did not use the expected transferred KV")
        if result["nixl_bytes"] <= 0 or result["nixl_ms_sum"] <= 0:
            raise ValueError("missing positive transfer measurements")
        if "kv_bytes_per_token" in config:
            block = config["block_size"]
            expected_bytes = count * (
                math.ceil((length - backoff) / block) * block * config["kv_bytes_per_token"]
                + config["state_bytes_per_request"]
            )
            if result["nixl_bytes"] != expected_bytes:
                raise ValueError("measured payload differs from the declared KV/state geometry")
            result["expected_payload_bytes"] = expected_bytes
    elif change(p, "vllm:request_success_total") != count:
        raise ValueError("direct prefill completion count differs")
    return result


def settle(config, before, count, routed):
    deadline = time.monotonic() + config["metrics_timeout"]
    p, d = (config[role]["label"] for role in ("prefill", "decode"))
    expected = [(p, PREFILL + "_count", count)]
    if routed:
        expected += [
            (d, name + "_count", count * config["tensor_parallel_size"])
            for name in (TIME, BYTES, POST)
        ]
        expected.append((d, "vllm:request_success_total", count))
    else:
        expected.append((p, "vllm:request_success_total", count))
    while True:
        after = io.snapshot(endpoints(config))
        differences = [io.delta(before, after, slot, name) for slot, name, _ in expected]
        if any(value > item[2] for value, item in zip(differences, expected)):
            raise ValueError("unrelated requests/transfers appeared in the measurement interval")
        if all(value == item[2] for value, item in zip(differences, expected)):
            return after
        if time.monotonic() >= deadline:
            raise ValueError("statistics publication did not match completed requests")
        time.sleep(0.25)


def trace_snapshot(config, before=None):
    endpoint = config["rank_trace_url"]
    if before:
        endpoint += "?" + urllib.parse.urlencode(before["counts"])
    with io.http(endpoint, timeout=10) as response:
        result = json.load(response)
    if not isinstance(result.get("counts"), dict) or not isinstance(result.get("rows"), list):
        raise ValueError("invalid rank trace response")
    if any(type(v) is not int or v < 0 for v in result["counts"].values()):
        raise ValueError("invalid rank trace offsets")
    if before and any(result["counts"].get(k, -1) < v for k, v in before["counts"].items()):
        raise ValueError("rank trace reset or file disappeared")
    return result


def rank_measurement(config, row, ranks):
    request_id = row["server_request_id"]
    if not request_id or len(ranks) != config["tensor_parallel_size"]:
        raise ValueError("expected one transfer per TP rank for the response request ID")
    if {r["rank"] for r in ranks} != set(range(config["tensor_parallel_size"])):
        raise ValueError("missing or duplicate transfer rank")
    domains = {r.get("clock_domain") for r in ranks}
    if len(domains) != 1 or (None in domains and not config.get("legacy_poster")):
        raise ValueError("TP rank clocks differ; request critical path requires one decode node")
    if any(
        not (r["request_id"] == request_id or r["request_id"].startswith(request_id + "-"))
        for r in ranks
    ):
        raise ValueError("rank trace belongs to a different response request ID")
    for rank in ranks:
        for name in ("posted_monotonic_ns", "observed_done_monotonic_ns", "bytes", "nixl_xfer_us"):
            io.positive(rank[name], "rank." + name)
        post = rank["nixl_post_us"]
        if not isinstance(post, (int, float)) or not math.isfinite(post) or post < 0:
            raise ValueError("invalid rank post timing")
        if rank["observed_done_monotonic_ns"] <= rank["posted_monotonic_ns"]:
            raise ValueError("invalid rank clock interval")
    first = min(r["posted_monotonic_ns"] for r in ranks)
    return {
        "nixl_critical_path_ms": (max(r["observed_done_monotonic_ns"] for r in ranks) - first)
        / 1e6,
        "rank_start_skew_ms": (max(r["posted_monotonic_ns"] for r in ranks) - first) / 1e6,
        "rank_max_xfer_ms": max(r["nixl_xfer_us"] for r in ranks) / 1000,
        "rank_payload_bytes": sum(r["bytes"] for r in ranks),
        "rank_xfer_ms_sum": sum(r["nixl_xfer_us"] for r in ranks) / 1000,
        "rank_post_ms_sum": sum(r["nixl_post_us"] for r in ranks) / 1000,
    }


def collect_ranks(config, old, rows, file, warmup=False):
    snapshot = trace_snapshot(config, old)
    remaining = list(snapshot["rows"])
    measured = {}
    for row in rows:
        request_id = row["server_request_id"]
        if not request_id:
            raise ValueError("missing response request ID for rank correlation")
        ranks = [
            r
            for r in remaining
            if r["request_id"] == request_id or r["request_id"].startswith(request_id + "-")
        ]
        remaining = [r for r in remaining if r not in ranks]
        measured[row["request"]] = rank_measurement(config, row, ranks)
        file.write(
            json.dumps(
                {
                    "request": row["request"],
                    "warmup": warmup,
                    "server_request_id": request_id,
                    "ranks": ranks,
                    **measured[row["request"]],
                },
                allow_nan=False,
            )
            + "\n"
        )
        file.flush()
    if remaining:
        raise ValueError("transfers outside the recorded requests appeared in rank traces")
    return measured


def rank_totals(metric, ranked):
    totals = {
        key: sum(r[key] for r in ranked.values())
        for key in ("rank_payload_bytes", "rank_xfer_ms_sum", "rank_post_ms_sum")
    }
    for trace, name in (
        ("rank_payload_bytes", "nixl_bytes"),
        ("rank_xfer_ms_sum", "nixl_ms_sum"),
        ("rank_post_ms_sum", "nixl_post_ms_sum"),
    ):
        if not math.isclose(totals[trace], metric[name], abs_tol=1e-5, rel_tol=1e-8):
            raise ValueError("rank traces differ from Prometheus: " + name)


def baseline(config):
    before = io.snapshot(endpoints(config))
    # Labels may not exist before an engine's first warmup observation.
    lazy_metrics = [SOURCE, "vllm:request_success_total"] + [
        name + suffix for name in (PREFILL, TIME, BYTES, POST) for suffix in ("_sum", "_count")
    ]
    for label, lines in before["pods"].items():
        for name in lazy_metrics:
            try:
                io.counter(before, label, name)
            except ValueError as exc:
                if not str(exc).startswith("missing metric"):
                    raise
                lines.append(name + " 0")
    return before


def measure(config, phase, length, concurrency, root):
    directory = root / f"{phase}-p{length}-c{concurrency}"
    directory.mkdir()
    count = config["aggregate_requests"] if phase == "aggregate" else config["runs"]
    cell = {
        **config,
        "phase": phase,
        "prompt_tokens": length,
        "concurrency": concurrency,
        "requests": count,
    }
    io.write_json(directory / "config.json", cell)
    routed = phase != "prefill"
    traced = routed and bool(config.get("rank_trace_url"))
    target = config["decode" if routed else "prefill"]
    extra = (
        None
        if routed
        else {"kv_transfer_params": {"do_remote_decode": False, "do_remote_prefill": False}}
    )
    # Paired serial prompts; aggregate and warmup prompts have separate keys.
    key = "aggregate" if phase == "aggregate" else "serial"
    try:
        prompts = [io.prepare(config, length, (key, concurrency, i)) for i in range(count)]
        io.save_prompts(directory, prompts)
        with (
            gzip.open(directory / "rank-transfers.jsonl.gz", "wt") as ranks_file,
            gzip.open(directory / "warmups.jsonl.gz", "wt") as warm_file,
        ):
            for i in range(config["warmup"]):
                before = baseline(config)
                old_trace = trace_snapshot(config) if traced else None
                prompt = io.prepare(config, length, (phase, concurrency, "warmup", i))
                row = io.send(
                    config,
                    target,
                    prompt,
                    -i - 1,
                    (phase, length, concurrency, "warmup", i),
                    1,
                    extra,
                )
                warm_file.write(json.dumps(row) + "\n")
                warm_file.flush()
                if not row["success"]:
                    raise ValueError("warmup failed; retained without retry")
                after = settle(config, before, 1, routed)
                metric = accounting(config, before, after, 1, length, routed)
                if traced:
                    ranked = collect_ranks(config, old_trace, [row], ranks_file, warmup=True)
                    rank_totals(metric, ranked)
            before = io.snapshot(endpoints(config))
            io.write_json(directory / "metrics-before.json", before)
            old_trace = trace_snapshot(config) if traced else None
            rows = []
            wall = None
            if phase == "aggregate":
                rows, wall = io.closed_loop(
                    concurrency,
                    prompts,
                    [target],
                    lambda t, p, i: io.send(config, t, p, i, (phase, length, concurrency, i), 1),
                    directory,
                )
                if not all(r["success"] for r in rows):
                    raise ValueError("measured requests failed; retained without retry")
            else:
                with (
                    gzip.open(directory / "requests.jsonl.gz", "wt") as raw,
                    gzip.open(directory / "request-metrics.jsonl.gz", "wt") as measurements,
                ):
                    for i, prompt in enumerate(prompts):
                        old = io.snapshot(endpoints(config))
                        trace_old = trace_snapshot(config) if traced else None
                        row = io.send(config, target, prompt, i, (phase, length, i), 1, extra)
                        rows.append(row)
                        raw.write(json.dumps(row) + "\n")
                        raw.flush()
                        if not row["success"]:
                            raise ValueError("measured request failed; retained without retry")
                        after = settle(config, old, 1, routed)
                        metric = accounting(config, old, after, 1, length, routed)
                        if traced:
                            ranked = collect_ranks(config, trace_old, [row], ranks_file)
                            rank_totals(metric, ranked)
                            metric.update(ranked[i])
                        elif routed:
                            metric["nixl_critical_path_ms"] = metric["nixl_ms_sum"]
                        measurements.write(
                            json.dumps(
                                {"request": i, "before": old, "after": after, "measurement": metric}
                            )
                            + "\n"
                        )
                        measurements.flush()
            after = settle(config, before, count, routed)
            metric = accounting(config, before, after, count, length, routed)
            if traced and phase == "aggregate":
                ranked = collect_ranks(config, old_trace, rows, ranks_file)
                rank_totals(metric, ranked)
            io.write_json(directory / "metrics-after.json", after)
            summary = {
                "model": config["model"],
                "transport": config["transport"],
                "phase": phase,
                "prompt_tokens": length,
                "concurrency": concurrency,
                "attempted": len(rows),
                "successful": sum(r["success"] for r in rows),
                **metric,
                "wall_seconds": wall,
                "aggregate_payload_GB_s": metric["nixl_bytes"] / wall / 1e9 if wall else None,
                "bandwidth_scope": SCOPE,
                "timing_scope": (
                    "backend prefill histogram"
                    if phase == "prefill"
                    else "common decode-node clock, post to observed completion"
                    if traced
                    else "TP1 backend telemetry"
                ),
            }
            io.write_json(directory / "summary.json", summary)
        io.write_json(directory / "complete.json", {"status": "complete"})
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        io.write_json(directory / "error.json", {"status": "invalid", "error": str(exc)})
        raise
    finally:
        io.manifest(directory)


def run(config, output):
    root = io.initialize(output, config, __file__)
    try:
        for length in config["prompt_lengths"]:
            for phase in ("prefill", "transfer"):
                measure(config, phase, length, 1, root)
        for concurrency in config["concurrency_levels"]:
            measure(config, "aggregate", config["aggregate_prompt_tokens"], concurrency, root)
        io.write_json(root / "campaign-complete.json", {"status": "complete"})
    except Exception as exc:
        io.write_json(root / "error.json", {"status": "incomplete", "error": str(exc)})
        raise
    finally:
        io.manifest(root)


def jsonl(path):
    with gzip.open(path, "rt") as file:
        return [json.loads(line) for line in file]


def legacy_config(config):
    return {
        **config,
        "legacy_poster": True,
        "tensor_parallel_size": config["tp"],
        "prefill": {"label": "bandwidth-prefill"},
        "decode": {"label": "bandwidth-decode"},
        "prefill_backoff_tokens": config["prompt_tokens"]
        - config.get("prefill_tokens", config["prompt_tokens"]),
    }


def inspect_cell(path, legacy):
    config = json.loads((path / "config.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    if legacy:
        config = legacy_config(config)
    else:
        io.verify_manifest(path)
        if not (path / "complete.json").is_file():
            raise ValueError(f"incomplete cell: {path}")
    phase, length = config["phase"], config["prompt_tokens"]
    rows = jsonl(path / "requests.jsonl.gz")
    if len(rows) != config["requests"] or {r["request"] for r in rows} != set(range(len(rows))):
        raise ValueError(f"missing or duplicate requests: {path}")
    if any(
        not r["success"]
        or r["usage"]["prompt_tokens"] != length
        or r["usage"]["completion_tokens"] != 1
        or len(r["generated_token_ids"]) != 1
        for r in rows
    ):
        raise ValueError(f"failed or incomplete requests: {path}")
    if summary["attempted"] != len(rows) or summary["successful"] != len(rows):
        raise ValueError(f"summary request count differs: {path}")
    if not legacy:
        prompts = jsonl(path / "prompts.jsonl.gz")
        hashes = {
            p["request"]: io.digest(p["token_ids"])
            for p in prompts
            if len(p["token_ids"]) == length
        }
        if (
            len(prompts) != len(rows)
            or set(hashes) != {r["request"] for r in rows}
            or any(hashes[r["request"]] != r["prompt_sha256"] for r in rows)
        ):
            raise ValueError(f"saved token-ID prompts differ from requests: {path}")
    before = json.loads((path / "metrics-before.json").read_text())
    after = json.loads((path / "metrics-after.json").read_text())
    measured = accounting(config, before, after, len(rows), length, phase != "prefill")
    for name, value in measured.items():
        if name not in summary or not math.isclose(
            value, summary[name], abs_tol=1e-5, rel_tol=1e-8
        ):
            raise ValueError(f"summary differs from counters: {path}/{name}")
    ranks_path = path / "rank-transfers.jsonl.gz"
    trace_rows = (
        [r for r in jsonl(ranks_path) if not r.get("warmup")] if ranks_path.exists() else []
    )
    traces = {r["request"]: r for r in trace_rows}
    if len(traces) != len(trace_rows):
        raise ValueError(f"duplicate rank trace request: {path}")
    ranked = {}
    if phase != "prefill" and config.get("rank_trace_url"):
        if set(traces) != {r["request"] for r in rows}:
            raise ValueError(f"rank trace coverage differs: {path}")
        ranked = {
            r["request"]: rank_measurement(config, r, traces[r["request"]]["ranks"]) for r in rows
        }
        rank_totals(measured, ranked)
    elif phase != "prefill" and config["tensor_parallel_size"] != 1:
        raise ValueError(f"TP > 1 without correlated traces: {path}")
    samples = []
    if phase != "aggregate":
        observations = jsonl(path / "request-metrics.jsonl.gz")
        if len(observations) != len(rows) or {m["request"] for m in observations} != {
            r["request"] for r in rows
        }:
            raise ValueError(f"per-request metric coverage differs: {path}")
        by_id = {r["request"]: r for r in rows}
        for observation in observations:
            i = observation["request"]
            metric = accounting(
                config, observation["before"], observation["after"], 1, length, phase != "prefill"
            )
            if phase != "prefill":
                if ranked:
                    rank_totals(metric, {i: ranked[i]})
                    metric.update(ranked[i])
                else:
                    metric["nixl_critical_path_ms"] = metric["nixl_ms_sum"]
            for name, value in metric.items():
                if not math.isclose(
                    value, observation["measurement"][name], abs_tol=1e-5, rel_tol=1e-8
                ):
                    raise ValueError(
                        f"per-request measurement differs from raw data: {path}/{i}/{name}"
                    )
            samples.append(
                {
                    "request": i,
                    "prompt_sha256": by_id[i]["prompt_sha256"],
                    "generated_token_ids": by_id[i]["generated_token_ids"],
                    **metric,
                }
            )
    if phase == "aggregate":
        io.positive(summary["wall_seconds"], "wall_seconds")
        measured["aggregate_payload_GB_s"] = measured["nixl_bytes"] / summary["wall_seconds"] / 1e9
        if not math.isclose(
            measured["aggregate_payload_GB_s"], summary["aggregate_payload_GB_s"], rel_tol=1e-8
        ):
            raise ValueError("aggregate payload rate differs from raw counters and wall time")
    return {
        "path": str(path),
        "config": config,
        "summary": summary,
        "samples": samples,
        "verified": measured,
    }


def bootstrap_ratio(pairs):
    rng = random.Random(260102)
    values = []
    for _ in range(1000):
        draw = rng.choices(pairs, k=len(pairs))
        values.append(sum(t for _, t in draw) / sum(p for p, _ in draw))
    return [io.percentile(values, 0.025), io.percentile(values, 0.975)]


def fit_payload(points):
    if len({p[0] for p in points}) < 2:
        return {"status": "insufficient distinct payload sizes"}
    xbar, ybar = (statistics.mean(p[i] for p in points) for i in (0, 1))
    slope = sum((x - xbar) * (y - ybar) for x, y in points) / sum(
        (x - xbar) ** 2 for x, _ in points
    )
    intercept = ybar - slope * xbar
    return {
        "status": "descriptive fit",
        "intercept_ms": intercept,
        "ms_per_GB": slope,
        "effective_payload_GB_s": 1000 / slope if slope > 0 else None,
        "physical_parameters": slope > 0 and intercept >= 0,
        "residuals_ms": [y - intercept - slope * x for x, y in points],
    }


def analyze(directories, output, legacy=False):
    cells = []
    for directory in directories:
        if not legacy:
            io.verify_manifest(directory)
            if not (directory / "campaign-complete.json").is_file():
                raise ValueError(f"incomplete campaign: {directory}")
        for path in sorted(directory.glob("*/summary.json")):
            phase = json.loads(path.read_text())["phase"]
            if phase in {"prefill", "transfer", "aggregate"}:
                cells.append(inspect_cell(path.parent, legacy))
    if not cells:
        raise ValueError("no completed bandwidth cells")

    def identity(cell):
        c = cell["config"]
        return (
            c["model"],
            c.get("model_revision"),
            c["transport"],
            c["tensor_parallel_size"],
            io.digest(c.get("deployment_settings", {})),
            c.get("nonce", c.get("seed")),
            c.get("prediction_commit"),
        )

    keyed = {}
    for cell in cells:
        c = cell["config"]
        key = (identity(cell), c["phase"], c["prompt_tokens"], c["concurrency"])
        if key in keyed:
            raise ValueError("duplicate bandwidth cell; separate independent campaigns by nonce")
        keyed[key] = cell
    comparisons = []
    points = {}
    for cell in cells:
        c = cell["config"]
        if c["phase"] != "transfer":
            continue
        key = identity(cell)
        prefill = keyed.get((key, "prefill", c["prompt_tokens"], 1))
        if not prefill:
            raise ValueError("transfer cell has no matching direct prefill baseline")
        left = {r["request"]: r for r in prefill["samples"]}
        right = {r["request"]: r for r in cell["samples"]}
        if set(left) != set(right) or any(
            left[i]["prompt_sha256"] != right[i]["prompt_sha256"] for i in left
        ):
            raise ValueError(
                "prefill and transfer requests do not use the same paired token-ID prompts"
            )
        pairs = [
            (left[i]["prefill_ms_sum"], right[i]["nixl_critical_path_ms"]) for i in sorted(left)
        ]
        prefill_ms = statistics.mean(p for p, _ in pairs)
        transfer_ms = statistics.mean(t for _, t in pairs)
        payload = statistics.mean(r["nixl_bytes"] for r in right.values())
        row = {
            "model": c["model"],
            "transport": c["transport"],
            "prompt_tokens": c["prompt_tokens"],
            "tensor_parallel_size": c["tensor_parallel_size"],
            "requests": len(pairs),
            "prefill_ms_mean": prefill_ms,
            "transfer_ms_mean": transfer_ms,
            "transfer_to_prefill_ratio": transfer_ms / prefill_ms,
            "ratio_bootstrap_95_interval": bootstrap_ratio(pairs) if len(pairs) >= 2 else None,
            "payload_bytes_mean": payload,
            "payload_GB_s_for_100pct_prefill": payload / (prefill_ms / 1000) / 1e9,
            "payload_GB_s_for_10pct_prefill": payload / (prefill_ms / 1000 * 0.1) / 1e9,
            "output_token_disagreements": sum(
                left[i]["generated_token_ids"] != right[i]["generated_token_ids"] for i in left
            ),
            "prefill_backoff_tokens": c.get("prefill_backoff_tokens", 0),
            "transport_verified": False,
            "transport_evidence": c.get("transport_evidence", {}),
            "scope": "payload-only thresholds omit fixed transfer overhead; transport label is a configuration claim",
            "prefill_path": prefill["path"],
            "transfer_path": cell["path"],
        }
        comparisons.append(row)
        points.setdefault(key, []).append((payload / 1e9, transfer_ms))
    root = pathlib.Path(output)
    root.mkdir(parents=True, exist_ok=False)
    io.write_json(root / "comparisons.json", comparisons)
    io.write_json(root / "cells.json", cells)
    fits = []
    for key, values in points.items():
        fit = fit_payload(values)
        port = next(c["config"].get("reported_port_GB_s") for c in cells if identity(c) == key)
        rate = fit.get("effective_payload_GB_s")
        fits.append(
            {
                "campaign": list(key),
                **fit,
                "reported_port_GB_s": port,
                "payload_fit_exceeds_reported_port": rate > port if rate and port else None,
                "scope": "payload timing fit; reported port comparison does not establish wire bandwidth",
            }
        )
    io.write_json(root / "fits.json", fits)
    io.write_json(
        root / "analysis-info.json",
        {
            "legacy_poster": legacy,
            "cells": len(cells),
            "comparisons": len(comparisons),
            "rank_clock_domain_ids_available": not legacy,
            "uncertainty_scope": "paired request bootstrap within each serial cell; excludes deployment/day variability",
            "timing_scope": "TP rank critical path includes completion observation delay; TP1 without traces uses backend telemetry",
            "bandwidth_scope": SCOPE,
        },
    )
    io.manifest(root)
    print(f"Verified {len(cells)} cells and {len(comparisons)} paired length comparisons in {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("config", type=pathlib.Path)
        if name == "run":
            sub.add_argument("--output", type=pathlib.Path, required=True)
    sub = commands.add_parser("analyze")
    sub.add_argument("directories", nargs="+", type=pathlib.Path)
    sub.add_argument("--output", type=pathlib.Path, required=True)
    sub.add_argument("--legacy-poster", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "analyze":
            analyze(args.directories, args.output, args.legacy_poster)
        else:
            config = validate(io.load_config(args.config))
            if args.command == "validate":
                print("Configuration valid")
            else:
                run(config, args.output)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"bandwidth: {exc}\n")


if __name__ == "__main__":
    main()
