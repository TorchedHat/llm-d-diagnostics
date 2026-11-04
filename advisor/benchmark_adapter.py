"""Attach request-level diagnostics to an llm-d benchmark v0.2 report.

The upstream benchmark report contains aggregates, so it is never converted into
per-request joint SLO attainment. An independently collected evidence attachment
is required before this adapter can return a measured decision.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from .evidence import VERSION, compare, sha256_file, verify_bundle_sources
except ImportError:
    from evidence import (  # type: ignore[no-redef]
        VERSION,
        compare,
        sha256_file,
        verify_bundle_sources,
    )


def recommend(report, attachment, *, report_sha256, sources_verified=False):
    """Consume a v0.2 mapping or an upstream Pydantic benchmark report object."""
    if hasattr(report, "model_dump"):
        report = report.model_dump(mode="json")
    if not isinstance(report, dict) or report.get("version") not in {"0.2", "0.2.1"}:
        return {"schema_version": VERSION, "status": "insufficient_evidence", "choice": None,
                "reasons": ["benchmark report version is missing or unsupported"]}
    if (not isinstance(attachment, dict) or not isinstance(attachment.get("benchmark_report_sha256"), list)
            or report_sha256 not in attachment["benchmark_report_sha256"]):
        return {"schema_version": VERSION, "status": "insufficient_evidence", "choice": None,
                "reasons": ["request evidence is not linked to this benchmark report"]}
    result = compare(attachment, artifacts_verified=sources_verified)
    return {"schema_version": VERSION, "status": "insufficient_evidence", "choice": None,
            "reasons": ["benchmark report and request evidence workload equivalence is not independently verified"],
            "benchmark_report": {"sha256": report_sha256, "version": report["version"],
                                 "aggregate_report_is_not_joint_slo_evidence": True},
            "request_evidence_decision": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Benchmark report v0.2 JSON or YAML")
    parser.add_argument("--attachment", required=True, help="Request-level evidence JSON")
    parser.add_argument("--output", required=True, help="New decision JSON file")
    args = parser.parse_args()
    try:
        from llmd_benchmark_report import import_benchmark_report
    except ImportError:
        print("Install the optional llm-d benchmark-report package to parse upstream reports.",
              file=sys.stderr)
        return 1
    try:
        report_path, attachment_path = Path(args.report), Path(args.attachment)
        report = import_benchmark_report(str(report_path))
        attachment = json.loads(attachment_path.read_text())
        result = recommend(report, attachment, report_sha256=sha256_file(report_path),
                           sources_verified=verify_bundle_sources(attachment, attachment_path.parent))
        result["attachment_sha256"] = sha256_file(attachment_path)
        with Path(args.output).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Cannot adapt benchmark report: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
