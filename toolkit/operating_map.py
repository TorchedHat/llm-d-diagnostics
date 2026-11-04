#!/usr/bin/env python3
"""Recompute an operating map from explicitly scoped per-cell summaries."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "llmd-diagnostics-operating-map/v1"


def load_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != VERSION:
        raise ValueError("unsupported operating-map configuration")
    arms, repeats = config.get("arms"), config.get("repeats")
    if (not isinstance(arms, list) or len(arms) < 2 or any(not isinstance(a, str) or not a for a in arms)
            or len(set(arms)) != len(arms)):
        raise ValueError("provide at least two distinct arm names")
    if (not isinstance(repeats, list) or len(repeats) < 2
            or any(type(r) is not int or r < 0 for r in repeats) or len(set(repeats)) != len(repeats)):
        raise ValueError("provide at least two distinct repeat identifiers")
    gpu = config.get("declared_gpus_per_arm")
    if type(gpu) is not int or gpu < 1:
        raise ValueError("declared_gpus_per_arm must be a positive integer")
    margin, gate = config.get("practical_margin"), config.get("min_observed_attainment")
    if (isinstance(margin, bool) or not isinstance(margin, (int, float))
            or not math.isfinite(margin) or not 0 <= margin < 1):
        raise ValueError("invalid practical margin")
    if gate is not None and (isinstance(gate, bool) or not isinstance(gate, (int, float))
                             or not math.isfinite(gate) or not 0 < gate <= 1):
        raise ValueError("invalid observed SLO attainment gate")
    return config


def winner(cells, arms, repeats, margin):
    if len(arms) < 2:
        return arms[0] if arms else "no_candidate_meets_slo"
    means = {arm: statistics.mean(cells[arm, r]["goodput_per_gpu"] for r in repeats) for arm in arms}
    ordered = sorted(arms, key=lambda arm: (-means[arm], arm))
    first, second = ordered[:2]
    relative = (means[first] - means[second]) / means[second] if means[second] > 0 else None
    consistent = all(cells[first, r]["goodput_per_gpu"] > cells[arm, r]["goodput_per_gpu"]
                     for r in repeats for arm in arms if arm != first)
    if consistent and (relative > margin if relative is not None else means[first] > 0):
        return first
    return "inconclusive"


def summarize(rows, config):
    config = load_config(config)
    arms, repeats, gpu = config["arms"], config["repeats"], config["declared_gpus_per_arm"]
    groups = defaultdict(dict)
    total_rows = 0
    for row in rows:
        if row.get("phase") != "main":
            continue
        try:
            model, revision, arm, load = (row[k] for k in ("model", "revision", "arm", "rate_key"))
            prompt, repeat = int(row["prompt_tokens"]), int(row["repeat"])
            offered, successful, failed, good = (int(row[k]) for k in
                                                 ("requests", "successes", "failures", "slo_passes"))
            wall, goodput = float(row["wall_seconds"]), float(row["goodput_per_gpu"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("missing or invalid cell fields") from exc
        if (not model or not revision or not load or prompt < 1 or arm not in arms or repeat not in repeats
                or row.get("valid", "").lower() != "true" or offered < 1 or successful < 0 or failed < 0
                or successful + failed != offered or not 0 <= good <= successful
                or not math.isfinite(wall) or wall <= 0 or not math.isfinite(goodput) or goodput < 0):
            raise ValueError("invalid or unplanned cell")
        if not math.isclose(goodput, good / wall / gpu, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("goodput differs from counts, wall time and declared GPUs")
        group, cell = (model, revision, prompt, load), (arm, repeat)
        if cell in groups[group]:
            raise ValueError("duplicate operating-map cell")
        groups[group][cell] = {"goodput_per_gpu": goodput, "offered": offered,
                               "successful": successful, "failed": failed, "slo_passes": good,
                               "observed_slo_attainment": good / offered, "wall_seconds": wall}
        total_rows += 1
    if not groups:
        raise ValueError("no main-phase cells")
    expected = {(arm, repeat) for arm in arms for repeat in repeats}
    positions = []
    for (model, revision, prompt, load), cells in sorted(groups.items()):
        if set(cells) != expected:
            raise ValueError("incomplete operating-map position")
        ungated = winner(cells, arms, repeats, config["practical_margin"])
        gate = config.get("min_observed_attainment")
        eligible = ([arm for arm in arms if all(cells[arm, repeat]["observed_slo_attainment"] >= gate
                                                for repeat in repeats)] if gate is not None else None)
        qualified = winner(cells, eligible, repeats, config["practical_margin"]) if eligible is not None else None
        positions.append({"model": model, "revision": revision, "prompt_tokens": prompt, "rate_key": load,
                          "relative_winner": ungated, "eligible_arms": eligible, "gated_winner": qualified,
                          "arms": {arm: [{"repeat": repeat, **cells[arm, repeat]} for repeat in repeats]
                                   for arm in arms}})
    return {"schema_version": VERSION, "config": config, "cells": total_rows, "positions": positions,
            "relative_winner_counts": dict(Counter(x["relative_winner"] for x in positions)),
            "scope": "Recomputed from cell summaries; raw request streams, prompt pairing, allocation and service quality are not independently verified."}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cells_csv")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="New JSON output file")
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text())
        with Path(args.cells_csv).open(newline="") as stream:
            result = summarize(csv.DictReader(stream), config)
        result["source_sha256"] = {"cells_csv": sha256(args.cells_csv), "config": sha256(args.config)}
        with Path(args.output).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"cells": result["cells"], "positions": len(result["positions"]),
                          "relative_winner_counts": result["relative_winner_counts"]}))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Cannot summarize operating map: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
