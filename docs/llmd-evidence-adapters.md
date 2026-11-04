# Attach an llm-d benchmark report

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
