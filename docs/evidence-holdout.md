# Freeze and check a measured choice

After importing and independently auditing the calibration evidence, freeze the
current choice before starting a separate profile with fresh inputs:

```bash
python3 advisor/evidence_holdout.py freeze --evidence /tmp/calibration/evidence.json \
  --output /tmp/frozen-advice.json
# Run and import a new profile with the same deployment and policy, using a new seed.
python3 advisor/evidence_holdout.py score --frozen /tmp/frozen-advice.json \
  --calibration /tmp/calibration/evidence.json --holdout /tmp/holdout/evidence.json \
  --output /tmp/holdout-result.json
```

The score checks recorded chronology, source and comparator hashes, matching
workload, policy, deployment fingerprints and controller/worker identities, plus
disjoint prompt hashes. It retains abstentions and compares the frozen choice
with the holdout decision. Results report each constant candidate's equal-block
objective regret against the observed block best, without filtering that baseline
by SLO eligibility. Per-cell calibration GPU seconds exclude warmups, idle time
and rollout. These samples and recorded timestamps do not independently establish
clock integrity, allocation or deployment equivalence. This is a same-deployment
holdout, not a prediction for unseen models, traffic mixes or clusters.
