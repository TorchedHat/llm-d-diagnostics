# Import a paired serving profile

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
