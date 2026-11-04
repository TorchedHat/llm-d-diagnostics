"""Experimental, side-effect-free provider for an explicitly configured planner."""

from pathlib import Path

try:
    from .evidence import VERSION, compare, verify_bundle_sources
except ImportError:
    from evidence import (  # type: ignore[no-redef]
        VERSION,
        compare,
        verify_bundle_sources,
    )


class EvidenceTopologyProvider:
    """Verify source files locally; caller-supplied booleans are not verification."""

    def __init__(self, evidence_base, allowed_root=None):
        self.base = Path(evidence_base).resolve()
        self.root = Path(allowed_root).resolve() if allowed_root is not None else self.base
        if not self.base.is_relative_to(self.root):
            raise ValueError("evidence base must be within the allowed root")

    def capabilities(self):
        return {"evidence_schemas": [VERSION], "objectives": ["ttft_p50", "ttft_p90",
                "joint_goodput_per_gpu"], "side_effects": False, "prediction": False}

    def recommend(self, context, candidates, evidence):
        def insufficient(reason):
            return {"schema_version": VERSION, "status": "insufficient_evidence",
                    "choice": None, "reasons": [reason]}

        if not isinstance(context, dict) or not isinstance(evidence, dict):
            return insufficient("context or evidence is missing")
        if context.get("workload") != evidence.get("workload") or context.get("policy") != evidence.get("policy"):
            return insufficient("requested workload or policy differs from evidence")
        recorded = evidence.get("candidates")
        if (not isinstance(candidates, list) or not all(isinstance(c, dict) for c in candidates)
                or not isinstance(recorded, list) or not all(isinstance(c, dict) for c in recorded)):
            return insufficient("candidate definitions are missing")
        keys = ("id", "topology", "comparison_fingerprint", "deployment_fingerprint", "allocation")
        try:
            requested = sorted(({k: c.get(k) for k in keys} for c in candidates), key=lambda c: c["id"])
            observed = sorted(({k: c.get(k) for k in keys} for c in recorded), key=lambda c: c["id"])
        except (TypeError, KeyError):
            return insufficient("candidate identifiers are invalid")
        if requested != observed:
            return insufficient("requested candidate definitions differ from measured deployments")
        sources = evidence.get("sources")
        if not isinstance(sources, list) or not sources:
            return insufficient("source artifacts are missing")
        try:
            for source in sources:
                if not isinstance(source, dict) or not isinstance(source.get("path"), str):
                    return insufficient("source artifact path is invalid")
                if not (self.base / source["path"]).resolve().is_relative_to(self.root):
                    return insufficient("source artifact is outside the allowed root")
            verified = verify_bundle_sources(evidence, self.base)
        except (OSError, ValueError):
            verified = False
        return compare(evidence, artifacts_verified=verified)
