# Measured topology evidence

`advisor/evidence.py` compares recorded request-level data. It does not deploy workloads,
measure hardware, predict unmeasured configurations, or verify model quality. The existing `advisor/plan.py` heuristic is a separate path and is not validated by this comparison.

The evidence attachment is JSON with `schema_version: "llmd-diagnostics-evidence/v1"`:

- `workload` contains `model_revision`, `tokenizer_revision`, input/output token counts,
  `load`, `sampling`, and a `comparison_fingerprint`. The fingerprint must cover common
  engine, serving, cache and request settings; placement and role split may differ by candidate.
- `comparison_verified` must be true only after the declared common settings and workload have
  been independently checked against the measured deployments and request artifacts.
- `policy` specifies `objective` (`ttft_p50`, `ttft_p90`, or `joint_goodput_per_gpu`),
  `ttft_slo_ms`, `tpot_slo_ms`, `min_observed_attainment`, `practical_margin`, `min_blocks`
  and `min_requests_per_block`. An optional `max_reserved_gpus` bounds each candidate.
  `require_quality` defaults to true; disabling it explicitly scopes the result to performance.
- `candidates` lists unique IDs, topology (`aggregated` or `disaggregated`), the common
  `comparison_fingerprint`, an exact `deployment_fingerprint`, `allocation` with `verified`,
  `exclusive_verified`, `active_gpus` and `reserved_gpus`,
  and `blocks`. When quality is required, `quality_status` must be `verified_pass`.
- Each block has a unique string `id`, `wall_seconds`, `source_sha256` and a `requests` list.
  Each request has a unique string `id`, `prompt_sha256`, `outcome` (`completed`, `failed`,
  `unknown`), and, when completed, finite `ttft_ms` and request-mean `tpot_ms`; `e2e_ms` is
  optional. All candidate
  arms must contain the same request IDs and prompt hashes in every paired block.
- `sources` lists `{ "path": "...", "sha256": "..." }` references relative to the evidence
  file. Each block's `source_sha256` refers to a normalized JSON file containing exactly its
  `candidate_id`, `comparison_fingerprint`, `block_id`, `wall_seconds` and `requests`.

Run `python3 advisor/evidence.py evidence.json --output decision.json`. The output reports
offered, completed, failed and unknown counts, p50/p90 TTFT and TPOT, end-to-end timing when
available for every completed request in a cell, and finite joint-SLO throughput per active and
reserved GPU. Per-cell GPU seconds count only the measured cell wall time, excluding warmup,
rollout and other idle cost; do not sum simultaneous reservations as if they were independent.
Every candidate must use the same active
and reserved GPU budget. A qualifying choice must meet the observed SLO fraction in every block;
multiple qualifying candidates must show the specified practical improvement consistently.
Otherwise it returns an explicit `insufficient_evidence`, `no_candidate_meets_slo` or
`inconclusive` status. The decision concerns these samples; even 100% in a small sample does
not establish a population-level SLO guarantee.

Hash verification detects changed files and checks that each normalized block matches its source
file. It does not prove the importer interpreted the original request logs correctly, or that a
declared allocation, model revision, quality result or comparison fingerprint is true. Preserve
the original raw records, audit importer transformations, and verify device claims and pod
identity independently before using an output to make a deployment decision. Sampling, quality,
transport witnesses and provenance have to be appropriate for the intended workload.


## Attach an llm-d benchmark report

`advisor/benchmark_adapter.py` accepts benchmark-report versions 0.2 and 0.2.1
through the optional upstream `llmd_benchmark_report` package. The attachment's
`benchmark_report_sha256` list must contain the hash of the report file. Aggregate
report percentiles cannot establish a request-level joint SLO.

```bash
python3 advisor/benchmark_adapter.py --report report.yaml \
  --attachment evidence.json --output benchmark-decision.json
```

The request evidence is scored separately and retained in
`request_evidence_decision`. The adapter returns `insufficient_evidence` as its
outer decision: a hash link does not establish that the report and attachment
represent the same workload or deployment. This is an experimental adapter, not a
registered upstream plugin or an automatic topology controller.

## Summarize stack discovery

```bash
python3 advisor/discovery_attachment.py stack-discovery.json --output discovery-inventory.json
```

This reads an llm-d stack-discovery default JSON export and preserves allowlisted
component identities and pod/sidecar status while omitting native environment,
URLs and annotations. Protect the original export. Errors produce a partial
inventory. The summary does not validate routing, allocation, GPU isolation or
model compatibility and does not authorize a deployment recommendation.
