"""Tests for transport.py: UCX_TLS parsing and the transport verdict."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import transport
from transport import cuda_ipc_allowed, overall, verdict


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
    def test_kv_on_pod_interface_is_tcp(self):
        # The live H100 observation: 511.7 MB NIXL, 525.7 MB on eth0.
        kind, reason = verdict(pair(511.7e6, 525.7e6), cap(), cap())
        self.assertEqual(kind, "tcp")
        self.assertIn("pod network (eth0)", reason)

    def test_host_network_tcp_names_host_network(self):
        kind, reason = verdict(pair(), cap(host_network=True), cap(host_network=True))
        self.assertEqual(kind, "tcp")
        self.assertIn("host network", reason)

    def test_kv_off_interface_with_roce_devices_is_rdma_roce(self):
        kind, _ = verdict(pair(100e6, 1e6), cap(rdma=True), cap(rdma=True))
        self.assertEqual(kind, "rdma-roce")

    def test_infiniband_link_layer(self):
        kind, _ = verdict(pair(100e6, 1e6), cap(rdma=True, link="InfiniBand"),
                          cap(rdma=True, link="InfiniBand"))
        self.assertEqual(kind, "rdma-ib")

    def test_rdma_needs_devices_on_both_sides(self):
        kind, _ = verdict(pair(100e6, 1e6), cap(rdma=False), cap(rdma=True))
        self.assertEqual(kind, "unexplained")

    def test_same_node_bypass_with_cuda_ipc_allowed(self):
        kind, _ = verdict(pair(100e6, 1e6, same_node=True), cap(), cap(cuda_ipc=True))
        self.assertEqual(kind, "cuda-ipc")

    def test_same_node_bypass_without_cuda_ipc_is_not_cuda_ipc(self):
        kind, _ = verdict(pair(100e6, 1e6, same_node=True), cap(), cap(cuda_ipc=False))
        self.assertEqual(kind, "unexplained")

    def test_partial_is_mixed(self):
        kind, _ = verdict(pair(100e6, 50e6), cap(), cap())
        self.assertEqual(kind, "mixed")

    def test_no_external_kv_means_no_transfer(self):
        kind, _ = verdict(pair(ext_tokens=0), cap(), cap())
        self.assertEqual(kind, "no-kv-transfer")
        kind, _ = verdict(pair(nixl_bytes=0, rx=5e6), cap(), cap())
        self.assertEqual(kind, "no-kv-transfer")

    def test_all_requests_failed(self):
        kind, _ = verdict(pair(ok=0), cap(), cap())
        self.assertEqual(kind, "error")

    def test_thresholds_are_the_documented_ones(self):
        self.assertEqual(transport.ON_NETDEV_MIN, 0.9)
        self.assertEqual(transport.OFF_NETDEV_MAX, 0.2)


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
                json.dump({"transport": "tcp", "scope": "cross-node",
                           "witnessed_at": "2026-10-11T00:00:00+0000", "pairs": []}, f)
            with patch.object(client, "DATA_DIR", d):
                self.assertEqual(client.witnessed_transport(),
                                 {"transport": "tcp", "scope": "cross-node",
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
                json.dump({"transport": "tcp", "scope": "cross-node", "witnessed_at": "t",
                           "pairs": [{"prefill_node": "w-v72kj", "decode_node": "w-s4fxf",
                                      "same_node": False, "transport": "tcp",
                                      "netdev_ratio": 1.03, "nixl_gbps": 0.31}]}, f)
            lines = analyze.transport_summary(d)
        self.assertEqual(lines[0], "KV transport: tcp (cross-node, witnessed t)")
        self.assertEqual(lines[1], "  v72kj -> s4fxf cross-node: tcp, "
                                   "interface/NIXL bytes 1.03, 0.31 GB/s")

    def test_header_flags_missing_witness(self):
        import tempfile

        import analyze
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(analyze.transport_summary(d)[0].startswith(
                "KV transport: NOT WITNESSED"))


if __name__ == "__main__":
    unittest.main()
