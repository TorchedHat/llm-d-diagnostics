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


## Import a paired serving profile

For each concurrency, convert a completed profile using explicit revisions and a
policy appropriate for your service. `examples/measured-policy.json` is an example,
not a recommended production SLO; its `require_quality: false` limits the decision
to observed performance.

```bash
python3 advisor/import_serving_profile.py /tmp/serving-profile \
  --output-dir /tmp/evidence-c8 --concurrency 8 \
  --model-revision "$MODEL_REVISION" --tokenizer-revision "$TOKENIZER_REVISION" \
  --policy examples/measured-policy.json
python3 advisor/evidence.py /tmp/evidence-c8/evidence.json \
  --output /tmp/evidence-c8/rechecked-decision.json
```

The importer writes `decision.json` and never overwrites an output directory. It
checks the raw hash, completed worker record, matching input prompts and successful
request completion/token counts. It attaches original and normalized file hashes.
An optional `--allocation-audit` is an external assertion; the importer does not
independently verify it. Declared revisions, exclusive GPU use and equivalent serving
configuration remain unverified, so imported profiles abstain. EPP router identity
is recorded, but Service membership, selected endpoints and application transfer
paths still require independent evidence. Audit original records and declarations
before adding verified assertions; a file hash proves integrity, not truth.
