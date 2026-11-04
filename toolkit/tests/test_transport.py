"""Tests for transport.py: UCX_TLS parsing and the transport verdict."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transport import counter_delta, cuda_ipc_allowed, overall, verdict


def cap(rdma=False, host_network=False, cuda_ipc=False, link="Ethernet"):
    return {
        "rdma_device": rdma,
        "host_network": host_network,
        "cuda_ipc_allowed": cuda_ipc,
        "hca_link_layers": {"mlx5_0": link} if rdma else {},
    }


def pair(nixl_bytes=100e6, rx=103e6, ext_tokens=2000, ok=5, same_node=False):
    return {
        "ok": ok,
        "same_node": same_node,
        "delta": {"nixl_bytes": nixl_bytes, "external_kv_tokens": ext_tokens},
        "netdev_ratio": rx / nixl_bytes if nixl_bytes > 0 else None,
        "busiest_netdev": "eth0",
    }


class TestCudaIpcAllowed(unittest.TestCase):
    def test_unset_allows_everything(self):
        self.assertTrue(cuda_ipc_allowed(""))

    def test_exclusion_list(self):
        self.assertFalse(cuda_ipc_allowed("^cuda_ipc"))
        self.assertFalse(cuda_ipc_allowed("^tcp,cuda_ipc"))
        self.assertTrue(cuda_ipc_allowed("^tcp"))

    def test_inclusion_list(self):
        self.assertTrue(cuda_ipc_allowed("rc,cuda"))
        self.assertTrue(cuda_ipc_allowed("rc,cuda_ipc"))
        self.assertFalse(cuda_ipc_allowed("rc,cuda_copy"))
        self.assertFalse(cuda_ipc_allowed("tcp"))


class TestVerdict(unittest.TestCase):
    def test_counter_ratios_and_devices_do_not_prove_transport(self):
        cases = [
            (pair(511.7e6, 525.7e6), cap(), cap()),
            (pair(100e6, 1e6), cap(rdma=True), cap(rdma=True)),
            (pair(100e6, 1e6, same_node=True), cap(), cap(cuda_ipc=True)),
        ]
        for evidence, prefill, decode in cases:
            self.assertEqual(verdict(evidence, prefill, decode)[0], "unknown")

    def test_no_external_kv_or_failed_requests(self):
        self.assertEqual(verdict(pair(nixl_bytes=0, ext_tokens=0), cap(), cap())[0], "no-kv-transfer")
        self.assertEqual(verdict(pair(ext_tokens=0), cap(), cap())[0], "unknown")
        self.assertEqual(verdict(pair(ok=0), cap(), cap())[0], "error")

    def test_counter_delta_rejects_missing_reset_and_nonfinite_values(self):
        self.assertEqual(counter_delta(2, 5), 3)
        for before, after in ((None, 1), (2, 1), (0, float("nan")), (0, float("inf")), (-1, 2)):
            self.assertIsNone(counter_delta(before, after))

    def test_missing_or_reset_counters_are_unknown(self):
        evidence = pair()
        evidence["delta"]["nixl_bytes"] = None
        self.assertEqual(verdict(evidence, cap(), cap())[0], "unknown")
        evidence = pair()
        evidence["invalidity"] = ["counter reset"]
        self.assertEqual(verdict(evidence, cap(), cap())[0], "unknown")


class TestOverall(unittest.TestCase):
    def test_cross_node_evidence_wins(self):
        pairs = [{"same_node": True, "transport": "cuda-ipc"},
                 {"same_node": False, "transport": "tcp"},
                 {"same_node": False, "transport": "tcp"}]
        self.assertEqual(overall(pairs), ("tcp", "cross-node"))

    def test_same_node_only_is_labelled(self):
        self.assertEqual(overall([{"same_node": True, "transport": "tcp"}]),
                         ("tcp", "same-node only"))

    def test_disagreement_is_reported(self):
        pairs = [{"same_node": False, "transport": "tcp"},
                 {"same_node": False, "transport": "rdma-roce"}]
        kind, _ = overall(pairs)
        self.assertEqual(kind, "inconsistent: rdma-roce, tcp")


class TestWitnessedTransport(unittest.TestCase):
    def test_reads_transport_json(self):
        import json
        import tempfile
        from unittest.mock import patch

        import client
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "transport.json"), "w") as f:
                json.dump({"classifier_version": 2, "transport": "unknown", "scope": "cross-node",
                           "witnessed_at": "2026-10-11T00:00:00+0000", "pairs": []}, f)
            with patch.object(client, "DATA_DIR", d):
                self.assertEqual(client.witnessed_transport(),
                                 {"transport": "unknown", "scope": "cross-node",
                                  "witnessed_at": "2026-10-11T00:00:00+0000"})

    def test_missing_witness_is_explicit(self):
        import tempfile
        from unittest.mock import patch

        import client
        with tempfile.TemporaryDirectory() as d, patch.object(client, "DATA_DIR", d):
            self.assertEqual(client.witnessed_transport()["transport"], "not witnessed")


class TestAnalyzeHeader(unittest.TestCase):
    def test_header_states_transport_and_pairs(self):
        import json
        import tempfile

        import analyze
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "transport.json"), "w") as f:
                json.dump({"classifier_version": 2, "transport": "unknown", "scope": "cross-node", "witnessed_at": "t",
                           "pairs": [{"prefill_node": "w-v72kj", "decode_node": "w-s4fxf",
                                      "same_node": False, "transport": "tcp",
                                      "netdev_ratio": 1.03, "nixl_gbps": 0.31}]}, f)
            lines = analyze.transport_summary(d)
        self.assertEqual(lines[0], "KV transport: unknown (cross-node, witnessed t)")
        self.assertEqual(lines[1], "  v72kj -> s4fxf cross-node: tcp, "
                                   "interface/NIXL bytes 1.03, 0.31 GB/s payload/recorded transfer time")

    def test_header_flags_missing_witness(self):
        import tempfile

        import analyze
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(analyze.transport_summary(d)[0].startswith(
                "KV transport: NOT WITNESSED"))


if __name__ == "__main__":
    unittest.main()
