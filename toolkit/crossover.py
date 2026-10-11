#!/usr/bin/env python3
"""Compare monolithic and P/D SLO goodput at equal declared GPU counts."""

import argparse
import csv
import gzip
import itertools
import json
import pathlib
import random
import re
import sys
import time

from experiment_io import (
    closed_loop,
    counter,
    delta,
    digest,
    initialize,
    load_config,
    manifest,
    percentile,
    positive,
    prepare,
    save_prompts,
    send,
    snapshot,
    url,
    verify_manifest,
    write_json,
)

SOURCE = "vllm:prompt_tokens_by_source_total"


def validate(config):
    for key, default in (("requests", 100), ("output_tokens", 256), ("repeats", 1)):
        config[key] = positive(config.get(key, default), key, integer=True)
    if config["output_tokens"] < 2:
        raise ValueError("at least two output tokens are required to measure mean ITL")
    if type(config.get("prefill_backoff_tokens", 0)) is not int or config.get(
        "prefill_backoff_tokens", 0
    ) not in (0, 1):
        raise ValueError("prefill_backoff_tokens must be 0 or 1")
    for key, default in (("slo_ttft_ms", 2000), ("slo_itl_ms", 50), ("metrics_timeout", 30)):
        config[key] = positive(config.get(key, default), key)
    for key, default in (("prompt_lengths", [512, 8000]), ("concurrency_levels", [1, 4, 16])):
        values = config.get(key, default)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} must be a nonempty list of distinct integers")
        config[key] = [positive(v, key, integer=True) for v in values]
        if len(set(config[key])) != len(config[key]):
            raise ValueError(f"{key} must contain distinct integers")
    if config["requests"] < 3 * max(config["concurrency_levels"]):
        raise ValueError("requests must be at least 3 times the largest concurrency")
    arms = config.get("arms", [])
    if len(arms) < 2 or {a.get("kind") for a in arms} != {"monolithic", "pd"}:
        raise ValueError("include a monolithic arm and at least one pd arm")
    names, counts = set(), set()
    for arm in arms:
        name = arm.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) or name in names:
            raise ValueError("arm names must be unique filename-safe strings")
        names.add(name)
        instances, targets = arm.get("instances", []), arm.get("targets", [])
        if not instances or not targets:
            raise ValueError(f"{name}: instances and targets are required")
        labels, metrics_urls, total_gpus, roles = set(), set(), 0, set()
        for instance in instances:
            label, role = instance.get("label"), instance.get("role")
            if not isinstance(label, str) or not label or label in labels:
                raise ValueError(f"{name}: instance labels must be unique")
            if role not in {"monolithic", "prefill", "decode"}:
                raise ValueError(f"{name}: invalid instance role")
            labels.add(label)
            roles.add(role)
            endpoint = url(instance.get("metrics_url", ""), "metrics_url")
            if endpoint in metrics_urls:
                raise ValueError(
                    "each instance needs its own metrics endpoint, not a shared Service"
                )
            metrics_urls.add(endpoint)
            total_gpus += positive(instance.get("gpus"), "instance.gpus", integer=True)
        expected = {"monolithic"} if arm["kind"] == "monolithic" else {"prefill", "decode"}
        if roles != expected:
            raise ValueError(f"{name}: roles must be {sorted(expected)}")
        for target in targets:
            url(target.get("url", ""), "target.url")
            if target.get("label") not in labels:
                raise ValueError("target label must identify its monolithic/decode instance")
            role = next(i["role"] for i in instances if i["label"] == target["label"])
            if role == "prefill":
                raise ValueError("P/D targets must point at decode sidecars")
            if arm["kind"] == "pd" and not target.get("headers", {}).get("x-prefiller-host-port"):
                raise ValueError("P/D targets need explicit x-prefiller-host-port routing")
        served = {t["label"] for t in targets}
        expected_targets = {i["label"] for i in instances if i["role"] != "prefill"}
        if served != expected_targets:
            raise ValueError(f"{name}: route to every serving instance")
        arm["gpu_count"] = total_gpus
        arm["prefill_backoff_tokens"] = (
            config.get("prefill_backoff_tokens", 0) if arm["kind"] == "pd" else 0
        )
        counts.add(total_gpus)
    if len(counts) != 1:
        raise ValueError("all arms must use the same total GPU count, including prefill GPUs")
    return config


def verify_accounting(arm, before, after, results):
    expected = sum(r["usage"]["prompt_tokens"] for r in results if r["success"])
    producers = [i for i in arm["instances"] if i["role"] != "decode"]
    receivers = [i for i in arm["instances"] if i["role"] == "decode"]
    successes = sum(r["success"] for r in results)
    prefill_expected = expected - successes * arm["prefill_backoff_tokens"]
    records = {}
    for instance in arm["instances"]:
        label = instance["label"]
        values = {
            s: delta(before, after, label, SOURCE, s)
            for s in ("local_compute", "local_cache_hit", "external_kv_transfer")
        }
        if values["local_cache_hit"]:
            raise ValueError(f"{label}: measured requests reused cached prompt tokens")
        records[label] = values
    computed = sum(records[i["label"]]["local_compute"] for i in producers)
    if computed != prefill_expected:
        raise ValueError(
            f"computed prompt tokens {computed} differ from successful-request prefill {prefill_expected}"
        )
    if arm["kind"] == "pd":
        received = sum(records[i["label"]]["external_kv_transfer"] for i in receivers)
        local = sum(records[i["label"]]["local_compute"] for i in receivers)
        if received < expected - successes or received > expected or local > successes:
            raise ValueError("decode did not use the expected transferred KV")
    return records


def account(arm, before, endpoints, results, timeout):
    expected = sum(r["usage"]["prompt_tokens"] for r in results if r["success"])
    successes = sum(r["success"] for r in results)
    prefill_expected = expected - successes * arm["prefill_backoff_tokens"]
    producers = [i for i in arm["instances"] if i["role"] != "decode"]
    receivers = [i for i in arm["instances"] if i["role"] == "decode"]
    deadline = time.monotonic() + timeout
    while True:
        after = snapshot(endpoints)
        try:
            observed = sum(delta(before, after, i["label"], SOURCE) for i in producers)
            received = sum(
                delta(before, after, i["label"], SOURCE, "external_kv_transfer") for i in receivers
            )
        except ValueError as exc:
            if not str(exc).startswith("missing metric") or time.monotonic() >= deadline:
                raise
            time.sleep(0.25)
            continue
        if observed >= prefill_expected and (not receivers or received >= expected - successes):
            return after, verify_accounting(arm, before, after, results)
        if time.monotonic() >= deadline:
            raise ValueError("prompt-source metrics did not publish the completed requests")
        time.sleep(0.25)


def measure(config, arm, length, concurrency, repeat, root):
    directory = root / f"{arm['name']}-p{length}-c{concurrency}-r{repeat}"
    cell = {
        **config,
        "arm": arm["name"],
        "prompt_tokens": length,
        "concurrency": concurrency,
        "repeat": repeat,
    }
    directory.mkdir()
    write_json(directory / "config.json", cell)
    try:
        key = (length, concurrency, repeat)
        prompts = [
            prepare(config, length, (*key, "measured", i)) for i in range(config["requests"])
        ]
        save_prompts(directory, prompts)
        endpoints = {i["label"]: i["metrics_url"] for i in arm["instances"]}
        warm_before = snapshot(endpoints)
        absent = []
        for label in endpoints:
            try:
                counter(warm_before, label, SOURCE)
            except ValueError as exc:
                if not str(exc).startswith("missing metric"):
                    raise
                # Lazy source registration is treated as zero only during initialization.
                warm_before["pods"][label].append(SOURCE + " 0")
                absent.append(label)
        write_json(
            directory / "warmups-before.json", {**warm_before, "initially_absent_sources": absent}
        )
        warm_rows = []
        with gzip.open(directory / "warmups.jsonl.gz", "wt") as warmups:
            for i in range(max(config["warmup"], len(arm["targets"]))):
                prompt = prepare(config, length, (*key, "warmup", i))
                row = send(
                    config,
                    arm["targets"][i % len(arm["targets"])],
                    prompt,
                    -i - 1,
                    (arm["name"], *key, "warmup", i),
                    config["output_tokens"],
                )
                warmups.write(json.dumps(row) + "\n")
                warmups.flush()
                warm_rows.append(row)
                if not row["success"]:
                    raise ValueError(f"warmup failed: {row['error']}")
        before, warm_account = account(
            arm, warm_before, endpoints, warm_rows, config["metrics_timeout"]
        )
        write_json(directory / "warmup-accounting.json", warm_account)
        write_json(directory / "metrics-before.json", before)

        def one(target, prompt, index):
            row = send(
                config,
                target,
                prompt,
                index,
                (arm["name"], *key, "measured", index),
                config["output_tokens"],
            )
            row["slo_good"] = (
                row["success"]
                and row["ttft_ms"] is not None
                and row["itl_mean_ms"] is not None
                and row["ttft_ms"] <= config["slo_ttft_ms"]
                and row["itl_mean_ms"] <= config["slo_itl_ms"]
            )
            return row

        rows, wall = closed_loop(concurrency, prompts, arm["targets"], one, directory)
        successful = [r for r in rows if r["success"]]
        summary = {
            "model": config["model"],
            "model_revision": config.get("model_revision"),
            "nonce": config["nonce"],
            "arm": arm["name"],
            "kind": arm["kind"],
            "prompt_tokens": length,
            "output_tokens": config["output_tokens"],
            "concurrency": concurrency,
            "repeat": repeat,
            "attempted": len(rows),
            "successful": len(successful),
            "slo_good": sum(r["slo_good"] for r in rows),
            "gpu_count": arm["gpu_count"],
            "wall_seconds": wall,
            "slo_ttft_ms": config["slo_ttft_ms"],
            "slo_itl_ms": config["slo_itl_ms"],
            "goodput_rps_per_gpu": sum(r["slo_good"] for r in rows) / wall / arm["gpu_count"],
            "request_rate_rps": len(successful) / wall,
            "itl_unavailable_requests": sum(r["itl_mean_ms"] is None for r in successful),
            "ttft_p50_ms": percentile([r["ttft_ms"] for r in successful], 0.5),
            "itl_mean_p50_ms": percentile([r["itl_mean_ms"] for r in successful], 0.5),
            "load_scope": "finite closed loop including ramp and drain",
            "gpu_count_source": "declared instance configuration",
            "deployment_settings": config.get("deployment_settings", {}),
            "status": "measured",
        }
        write_json(directory / "summary.json", summary)
        after, records = account(arm, before, endpoints, rows, config["metrics_timeout"])
        write_json(directory / "metrics-after.json", after)
        write_json(directory / "accounting.json", records)
        write_json(directory / "complete.json", {"requests": len(rows)})
        return summary
    except Exception as exc:
        write_json(directory / "error.json", {"error": str(exc), "status": "invalid"})
        raise
    finally:
        manifest(directory)


def run(config, output, selected=None):
    arms = config["arms"]
    if selected:
        names = set(selected.split(","))
        if not names <= {a["name"] for a in arms}:
            raise ValueError("--arms contains an unknown arm")
        arms = [a for a in arms if a["name"] in names]
    root = initialize(output, config, __file__)
    try:
        cells = list(
            itertools.product(
                arms,
                config["prompt_lengths"],
                config["concurrency_levels"],
                range(1, config["repeats"] + 1),
            )
        )
        random.Random(config["nonce"]).shuffle(cells)
        summaries = []
        for arm, length, concurrency, repeat in cells:
            print(
                f"{arm['name']}: prompt={length} concurrency={concurrency} repeat={repeat}",
                flush=True,
            )
            summaries.append(measure(config, arm, length, concurrency, repeat, root))
        write_json(
            root / "campaign-complete.json",
            {
                "cells": len(summaries),
                "selected_arms": [a["name"] for a in arms],
                "full_arm_matrix": len(arms) == len(config["arms"]),
            },
        )
        return summaries
    except Exception as exc:
        write_json(root / "error.json", {"error": str(exc), "status": "incomplete"})
        raise
    finally:
        manifest(root)


def analyze(directories, output, legacy=False):
    summaries, identities = [], set()
    for root in directories:
        if not legacy:
            verify_manifest(root)
            if not (root / "campaign-complete.json").exists():
                raise ValueError(f"incomplete campaign: {root}")
        for path in sorted(root.glob("*/summary.json")):
            if not (path.parent / "complete.json").exists():
                raise ValueError(f"incomplete cell: {path.parent}")
            row = json.loads(path.read_text())
            if legacy and row.get("phase") != "main":
                continue
            with gzip.open(path.parent / "requests.jsonl.gz", "rt") as file:
                requests = [json.loads(line) for line in file]
            if len(requests) != row["attempted"] or len({r["request"] for r in requests}) != len(
                requests
            ):
                raise ValueError(f"request count or identity mismatch: {path}")
            if any(
                r["success"]
                and (
                    r["usage"]["prompt_tokens"] != row["prompt_tokens"]
                    or r["usage"]["completion_tokens"] != row["output_tokens"]
                    or len(r["generated_token_ids"]) != row["output_tokens"]
                )
                for r in requests
            ):
                raise ValueError(f"request token count mismatch: {path}")
            if not legacy:
                config = json.loads((path.parent / "config.json").read_text())
                arm = next(a for a in config["arms"] if a["name"] == row["arm"])
                before = json.loads((path.parent / "metrics-before.json").read_text())
                after = json.loads((path.parent / "metrics-after.json").read_text())
                accounting = verify_accounting(arm, before, after, requests)
                if accounting != json.loads((path.parent / "accounting.json").read_text()):
                    raise ValueError(f"saved accounting differs from raw counters: {path}")
                with gzip.open(path.parent / "prompts.jsonl.gz", "rt") as file:
                    prompts = [json.loads(line) for line in file]
                if len(prompts) != len(requests) or {p["request"] for p in prompts} != {
                    r["request"] for r in requests
                }:
                    raise ValueError(f"prompt coverage mismatch: {path}")
                hashes = {
                    p["request"]: digest(p["token_ids"])
                    for p in prompts
                    if len(p["token_ids"]) == row["prompt_tokens"]
                }
                if any(hashes.get(r["request"]) != r["prompt_sha256"] for r in requests):
                    raise ValueError(f"prompt hash mismatch: {path}")
            good = sum(
                r["success"]
                and r.get("ttft_ms") is not None
                and r.get("itl_mean_ms") is not None
                and r["ttft_ms"] <= row["slo_ttft_ms"]
                and r["itl_mean_ms"] <= row["slo_itl_ms"]
                for r in requests
            )
            if good != row["slo_good"] or sum(r["success"] for r in requests) != row["successful"]:
                raise ValueError(f"SLO or success count mismatch: {path}")
            expected = (
                good
                / positive(row["wall_seconds"], "wall_seconds")
                / positive(row["gpu_count"], "gpu_count")
            )
            if abs(expected - row["goodput_rps_per_gpu"]) > 1e-9:
                raise ValueError(f"goodput does not match retained requests: {path}")
            identity = (
                row["model"],
                row["arm"],
                row["prompt_tokens"],
                row["concurrency"],
                row.get("repeat", 1),
            )
            if identity in identities:
                raise ValueError(f"duplicate cell: {identity}")
            identities.add(identity)
            summaries.append(
                {
                    **row,
                    "directory": str(path.parent),
                    "prompt_hashes": {str(r["request"]): r["prompt_sha256"] for r in requests},
                }
            )
    if not summaries:
        raise ValueError("no completed crossover cells")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "cells.json", summaries)
    comparisons = []
    for mono in summaries:
        if (
            mono.get("kind", "monolithic" if mono["arm"].startswith("MONO") else "pd")
            != "monolithic"
        ):
            continue
        for pd in summaries:
            if pd.get("kind", "monolithic" if pd["arm"].startswith("MONO") else "pd") != "pd":
                continue
            keys = (
                "model",
                "model_revision",
                "prompt_tokens",
                "output_tokens",
                "concurrency",
                "repeat",
                "slo_ttft_ms",
                "slo_itl_ms",
                "deployment_settings",
                "nonce",
            )
            if pd["arm"] == mono["arm"] or any(pd.get(k) != mono.get(k) for k in keys):
                continue
            if pd["gpu_count"] != mono["gpu_count"]:
                raise ValueError("comparison has unequal GPU counts")
            if not legacy and pd["prompt_hashes"] != mono["prompt_hashes"]:
                raise ValueError("comparison does not use paired token-ID prompts")
            comparisons.append(
                {
                    "model": mono["model"],
                    "prompt_tokens": mono["prompt_tokens"],
                    "concurrency": mono["concurrency"],
                    "repeat": mono.get("repeat", 1),
                    "monolithic_arm": mono["arm"],
                    "pd_arm": pd["arm"],
                    "monolithic_goodput": mono["goodput_rps_per_gpu"],
                    "pd_goodput": pd["goodput_rps_per_gpu"],
                    "pd_wins": pd["goodput_rps_per_gpu"] > mono["goodput_rps_per_gpu"],
                }
            )
    write_json(output / "comparisons.json", comparisons)
    if comparisons:
        with (output / "comparisons.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    write_json(
        output / "analysis-info.json",
        {
            "legacy_poster_input": legacy,
            "cells": len(summaries),
            "comparisons": len(comparisons),
            "interpretation": "sampled operating points; later reversals retained; no statistically established crossover threshold",
            "gpu_count": "declared total across prefill and decode; independently verify deployment resources",
        },
    )
    manifest(output)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "validate"):
        child = commands.add_parser(name)
        child.add_argument("config", type=pathlib.Path)
        if name == "run":
            child.add_argument("--output", type=pathlib.Path, required=True)
            child.add_argument(
                "--arms", help="comma-separated arm names for sequential deployments"
            )
    child = commands.add_parser("analyze")
    child.add_argument("directories", nargs="+", type=pathlib.Path)
    child.add_argument("--output", type=pathlib.Path, required=True)
    child.add_argument(
        "--legacy-poster", action="store_true", help="explicitly accept original poster cell format"
    )
    args = parser.parse_args()
    try:
        if args.command == "analyze":
            comparisons = analyze(args.directories, args.output, args.legacy_poster)
            print(f"Compared {len(comparisons)} sampled operating points; artifacts: {args.output}")
        else:
            config = validate(load_config(args.config))
            if args.command == "validate":
                print("Configuration valid: equal GPU counts and explicit routing")
            else:
                run(config, args.output, args.arms)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
