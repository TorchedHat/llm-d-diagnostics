from advisor.benchmark_adapter import recommend
from advisor.tests.test_evidence import bundle


def test_aggregate_report_cannot_create_joint_slo_evidence():
    result = recommend({"version": "0.2.1", "results": {}}, None,
                       report_sha256="a" * 64, sources_verified=True)
    assert result["status"] == "insufficient_evidence"


def test_linked_request_evidence_is_scored_without_claiming_report_equivalence():
    attachment = bundle()
    attachment["benchmark_report_sha256"] = ["a" * 64]
    result = recommend({"version": "0.2.1"}, attachment,
                       report_sha256="a" * 64, sources_verified=True)
    assert result["status"] == "insufficient_evidence" and result["choice"] is None
    assert result["request_evidence_decision"]["status"] == "measured"
    assert result["request_evidence_decision"]["choice"] == "pd"
    assert result["benchmark_report"]["aggregate_report_is_not_joint_slo_evidence"]


def test_report_hash_link_is_required():
    attachment = bundle()
    attachment["benchmark_report_sha256"] = ["b" * 64]
    result = recommend({"version": "0.2.1"}, attachment,
                       report_sha256="a" * 64, sources_verified=True)
    assert result["status"] == "insufficient_evidence"
    assert "not linked" in result["reasons"][0]
