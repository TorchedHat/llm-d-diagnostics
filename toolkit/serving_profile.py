#!/usr/bin/env python3
"""Run an equal-GPU, paired serving calibration against existing pods."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from hardware_profile import cluster_server, collect, pod_exec, pod_metadata, save


def valid_epp_url(value):
    try:
        parsed = urlsplit(value)
        return (parsed.scheme in {"http", "https"} and bool(parsed.hostname) and parsed.username is None
                and parsed.password is None and not parsed.query and not parsed.fragment
                and parsed.path == "/v1/completions" and (parsed.port is None or 1 <= parsed.port <= 65535))
    except ValueError:
        return False


def validate_events(events, concurrencies, blocks, requests, input_tokens):
    """Require every declared cell and paired request before marking a run complete."""
    if (not events or any(not isinstance(event, dict) for event in events)
            or events[-1].get("event") != "complete"
            or sum(event.get("event") == "complete" for event in events) != 1):
        raise ValueError("worker did not report completion")
    keys = {(arm, concurrency, block) for arm in ("MONO", "PD")
            for concurrency in concurrencies for block in range(blocks)}
    cells, rows, inputs, warmups = {}, {}, {}, set()
    for event in events:
        kind = event.get("event")
        if kind == "inputs":
            key = (event.get("concurrency"), event.get("block"))
            prompts = event.get("tokens")
            if (any(type(value) is not int for value in key) or key in inputs
                    or not isinstance(prompts, list) or len(prompts) != requests
                    or any(not isinstance(p, list) or len(p) != input_tokens
                           or any(type(token) is not int or token < 0 for token in p) for p in prompts)):
                raise ValueError("invalid or duplicate paired input set")
            inputs[key] = [hashlib.sha256(json.dumps(prompt).encode()).hexdigest() for prompt in prompts]
        elif kind == "cell":
            key = (event.get("arm"), event.get("concurrency"), event.get("block"))
            wall = event.get("wall_seconds")
            if (not isinstance(key[0], str) or any(type(value) is not int for value in key[1:])
                    or key not in keys or key in cells or event.get("requests") != requests
                    or type(event.get("success")) is not int or not 0 <= event["success"] <= requests
                    or isinstance(wall, bool) or not isinstance(wall, (int, float))
                    or not math.isfinite(wall) or wall <= 0):
                raise ValueError("invalid or duplicate serving cell")
            cells[key] = event
        elif kind == "request":
            arm, number = event.get("arm"), event.get("number")
            if event.get("warmup") is True:
                key = (arm, number)
                if (arm not in ("MONO", "PD") or type(number) is not int or number not in (0, 1)
                        or key in warmups or event.get("ok") is not True):
                    raise ValueError("invalid or duplicate warmup")
                warmups.add(key)
            elif event.get("warmup") is False:
                key = (arm, event.get("concurrency"), event.get("block"))
                if (not isinstance(arm, str) or any(type(value) is not int for value in key[1:])
                        or key not in keys or type(number) is not int or not 0 <= number < requests
                        or type(event.get("ok")) is not bool or (key, number) in rows):
                    raise ValueError("invalid or duplicate measured request")
                rows[key, number] = event
            else:
                raise ValueError("request warmup status is missing")
    if set(cells) != keys or len(warmups) != 4 or len(rows) != len(keys) * requests:
        raise ValueError("missing serving cells, warmups or measured requests")
    if set(inputs) != {(c, b) for c in concurrencies for b in range(blocks)}:
        raise ValueError("missing or unexpected paired input set")
    for key, cell in cells.items():
        if cell["success"] != sum(rows[key, i]["ok"] for i in range(requests)):
            raise ValueError("cell success count differs from request records")
        if any(rows[key, i].get("prompt_sha256") != inputs[key[1], key[2]][i] for i in range(requests)):
            raise ValueError("request prompt differs from its paired input")
    return list(cells.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--namespace", required=True)
    p.add_argument("--client-pod", required=True)
    p.add_argument("--client-container", default="")
    p.add_argument("--serving-container", default="", help="Container holding vLLM and its GPU")
    p.add_argument("--mono-pods", required=True, help="Two comma-separated TP1 monolithic pods")
    p.add_argument("--prefill-pod", required=True)
    p.add_argument("--decode-pod", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cli", choices=["oc", "kubectl"], default="oc")
    p.add_argument("--mono-port", type=int, default=8000)
    p.add_argument("--prefill-port", type=int, default=8000)
    p.add_argument("--sidecar-port", type=int, default=8000)
    p.add_argument("--pd-route", choices=["manual", "epp"], default="manual")
    p.add_argument("--epp-url", default="", help="Standalone/router HTTP completions endpoint for EPP mode")
    p.add_argument("--router-pod", default="", help="Router pod whose identity and restarts are tracked")
    p.add_argument("--input-tokens", type=int, default=4096)
    p.add_argument("--output-tokens", type=int, default=64)
    p.add_argument("--concurrencies", default="2,8")
    p.add_argument("--requests", type=int, default=32)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261104)
    p.add_argument("--timeout-seconds", type=int, default=180)
    args = p.parse_args()
    mono = [name.strip() for name in args.mono_pods.split(",") if name.strip()]
    names = [*mono, args.prefill_pod, args.decode_pod]
    try:
        concurrencies = [int(c) for c in args.concurrencies.split(",")]
        if (len(mono) != 2 or len(set(names)) != 4 or not concurrencies
                or len(set(concurrencies)) != len(concurrencies)
                or min(concurrencies) < 1 or max(concurrencies) > args.requests
                or not 1 <= args.blocks <= 20 or not 8 <= args.requests <= 512
                or not 1 <= args.input_tokens <= 131072 or not 1 <= args.output_tokens <= 4096
                or not 1 <= args.timeout_seconds <= 3600
                or (args.pd_route == "epp" and (not valid_epp_url(args.epp_url)
                    or not args.router_pod or args.router_pod in names or args.router_pod == args.client_pod))
                or (args.pd_route == "manual" and (args.epp_url or args.router_pod))
                or any(not 1 <= port <= 65535 for port in
                       (args.mono_port, args.prefill_port, args.sidecar_port))):
            raise ValueError
    except ValueError:
        p.error("invalid pods, workload, ports or bounds")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        inventory = collect(args.cli, args.namespace, names, args.serving_container)
        pods = {pod["name"]: pod for pod in inventory["pods"]}
        client = pod_metadata(args.cli, args.namespace, args.client_pod)
        router = pod_metadata(args.cli, args.namespace, args.router_pod) if args.pd_route == "epp" else None
        if (not client["ready"] or (router is not None and (not router["ready"] or not router["ip"]))
                or any(not pod["ready"] or not pod["ip"] for pod in pods.values())):
            raise ValueError("client and serving pods must be ready")
        uuids = []
        for pod in pods.values():
            gpu_query = pod["inventory"]["gpus"]
            lines = gpu_query["stdout"].splitlines()
            if gpu_query["status"] != 0 or len(lines) != 1:
                raise ValueError("each serving container must expose exactly one GPU")
            fields = lines[0].split(",")
            if len(fields) < 2 or not fields[1].strip():
                raise ValueError("GPU inventory is missing a device UUID")
            uuids.append(fields[1].strip())
        if len(set(uuids)) != 4:
            raise ValueError("serving containers must expose four distinct GPUs")
        config = {
            "model": args.model, "mono_urls": [f"http://{pods[n]['ip']}:{args.mono_port}/v1/completions"
                                               for n in mono],
            "pd_url": (args.epp_url if args.pd_route == "epp" else
                       f"http://{pods[args.decode_pod]['ip']}:{args.sidecar_port}/v1/completions"),
            "pd_route": args.pd_route,
            "prefill_host": f"{pods[args.prefill_pod]['ip']}:{args.prefill_port}",
            "input_tokens": args.input_tokens, "output_tokens": args.output_tokens,
            "concurrencies": concurrencies, "requests": args.requests, "blocks": args.blocks,
            "seed": args.seed, "timeout_seconds": args.timeout_seconds,
        }
        source = (Path(__file__).parent / "serving_profile_worker.py").read_text()
        metadata = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                    "cluster": cluster_server(args.cli),
                    "namespace": args.namespace, "client": client, "pods": pods, "router": router,
                    "roles": {"mono": mono, "prefill": args.prefill_pod, "decode": args.decode_pod},
                    "gpu_counts": {"MONO": 2, "PD": 2}, "config": config,
                    "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "worker_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "scope": f"{args.pd_route} P/D routing, sequential cells, closed-loop finite traffic; "
                             "operator must verify routing, allocation, model and TP settings."}
        save(out / "metadata.json", metadata)
        command = pod_exec(args.cli, args.namespace, args.client_pod, args.client_container, source)
        with (out / "raw.jsonl").open("w") as stdout, (out / "worker.log").open("w") as stderr:
            result = subprocess.run([*command, json.dumps(config)], stdout=stdout, stderr=stderr,
                                    check=False)
        events = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
        if result.returncode:
            raise ValueError(f"worker failed or incomplete (exit {result.returncode}); inspect raw files")
        cells = validate_events(events, concurrencies, args.blocks, args.requests, args.input_tokens)
        observed = {**pods, args.client_pod: client}
        if router is not None:
            observed[args.router_pod] = router
        for name, before in observed.items():
            after = pod_metadata(args.cli, args.namespace, name)
            if (after["uid"] != before["uid"] or not after["ready"]
                    or after["image_ids"] != before["image_ids"]
                    or after["restart_counts"] != before["restart_counts"]):
                raise ValueError(f"pod identity, image or readiness changed: {name}")
        save(out / "result.json", {"schema_version": 1, "metadata": metadata, "cells": cells,
                                   "raw_sha256": hashlib.sha256((out / "raw.jsonl").read_bytes()).hexdigest(),
                                   "status": "complete"})
        print(f"Serving profile written to {out / 'result.json'}")
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        save(out / "error.json", {"error": str(exc), "status": "incomplete"})
        print(f"Serving profile incomplete: {exc}; evidence retained in {out}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
