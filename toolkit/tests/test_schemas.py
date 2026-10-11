"""Tests for schemas.py: row schemas, config enums, fields_for, and TypedCSVWriter round-trips.

Covers every TypedDict row type with a write-read-back cycle to catch schema drift.
"""

import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from schemas import (
    CacheHit,
    CachePhase,
    CacheState,
    ConfigDecompose,
    ConfigExp1,
    ConfigIsolation,
    ConfigMixed,
    ConfigPaired,
    ConfigThroughput,
    EvictionPhase,
    Exp1bRow,
    Exp1Row,
    Exp2Row,
    Exp3Row,
    Exp4Row,
    Exp5bRow,
    Exp5Row,
    Exp6Row,
    Exp7GpuRow,
    Exp7Row,
    Exp8Row,
    Exp9Row,
    Exp10Row,
    Exp11Row,
    Exp12Row,
    Exp14Row,
    Exp16Row,
    Exp18Row,
    ModelLoadPhase,
    Priority,
    TypedCSVWriter,
    Weight,
    WorkloadClass,
    fields_for,
    priority_for,
)


class TestFieldsFor(unittest.TestCase):

    def test_returns_list_of_strings(self):
        fields = fields_for(Exp1Row)
        self.assertIsInstance(fields, list)
        for f in fields:
            self.assertIsInstance(f, str)

    def test_preserves_order(self):
        fields = fields_for(Exp1Row)
        self.assertEqual(fields[0], "experiment")
        self.assertEqual(fields[1], "config")
        self.assertEqual(fields[-1], "error")

    def test_all_row_types_have_experiment_field(self):
        row_types = [
            Exp1Row, Exp1bRow, Exp2Row, Exp3Row, Exp4Row, Exp5Row, Exp5bRow,
            Exp6Row, Exp7Row, Exp7GpuRow, Exp8Row, Exp9Row, Exp10Row,
            Exp11Row, Exp12Row, Exp14Row, Exp16Row, Exp18Row,
        ]
        for rt in row_types:
            fields = fields_for(rt)
            self.assertIn("experiment", fields, f"{rt.__name__} missing 'experiment' field")

    def test_field_count_stability(self):
        self.assertEqual(len(fields_for(Exp1Row)), 12)
        self.assertEqual(len(fields_for(Exp11Row)), 13)
        self.assertEqual(len(fields_for(Exp14Row)), 12)
        self.assertEqual(len(fields_for(Exp12Row)), 18)
        self.assertEqual(len(fields_for(Exp16Row)), 17)
        self.assertEqual(len(fields_for(Exp18Row)), 11)


class TestConfigEnums(unittest.TestCase):

    def test_value_serialization(self):
        self.assertEqual(ConfigExp1.BASELINE.value, "BASELINE")
        self.assertEqual(ConfigDecompose.A_PREFILL_DIRECT.value, "A-prefill-direct")
        self.assertEqual(ConfigThroughput.DISAGG_1D.value, "DISAGG-1D")

    def test_csv_compatible(self):
        self.assertEqual(ConfigExp1.BASELINE, "BASELINE")
        self.assertEqual(ConfigDecompose.D_DISAGGREGATED, "D-disaggregated")

    def test_config_paired_aliases(self):
        self.assertIs(ConfigIsolation, ConfigPaired)
        self.assertIs(ConfigMixed, ConfigPaired)

    def test_weight_values(self):
        self.assertEqual(Weight.HEAVY, "heavy")
        self.assertEqual(Weight.LIGHT, "light")

    def test_priority_for(self):
        self.assertEqual(priority_for(WorkloadClass.SHORT), Priority.HIGH)
        self.assertEqual(priority_for(WorkloadClass.LONG), Priority.LOW)

    def test_cache_phase_values(self):
        phases = [p.value for p in CachePhase]
        self.assertIn("hit_miss", phases)
        self.assertIn("multi_turn", phases)
        self.assertIn("decay", phases)

    def test_cache_state_values(self):
        self.assertEqual(CacheState.COLD, "cold")
        self.assertEqual(CacheState.WARM_SAME_POD, "warm_same_pod")

    def test_model_load_phase_completeness(self):
        phases = {p.value for p in ModelLoadPhase}
        self.assertIn("pod_delete", phases)
        self.assertIn("total_startup", phases)
        self.assertIn("wall_total", phases)

    def test_eviction_phase_completeness(self):
        phases = {p.value for p in EvictionPhase}
        self.assertIn("dist_cold", phases)
        self.assertIn("pressure_after", phases)

    def test_cache_hit_values(self):
        self.assertEqual(CacheHit.YES, "yes")
        self.assertEqual(CacheHit.NO, "no")
        self.assertEqual(CacheHit.NA, "n/a")


def _make_row(row_type):
    """Create a dummy row with all fields set to their field name."""
    return {f: f for f in fields_for(row_type)}


class TestTypedCSVWriterRoundTrip(unittest.TestCase):
    """Write-read-back for every row type. Catches schema drift."""

    ROW_TYPES = [
        Exp1Row, Exp1bRow, Exp2Row, Exp3Row, Exp4Row, Exp5Row, Exp5bRow,
        Exp6Row, Exp7Row, Exp7GpuRow, Exp8Row, Exp9Row, Exp10Row,
        Exp11Row, Exp12Row, Exp14Row, Exp16Row, Exp18Row,
    ]

    def test_round_trip_all_row_types(self):
        for row_type in self.ROW_TYPES:
            with self.subTest(row_type=row_type.__name__):
                self._round_trip(row_type)

    def _round_trip(self, row_type):
        row = _make_row(row_type)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            path = f.name

        try:
            writer = TypedCSVWriter(path, row_type)
            writer.write(row)
            writer.close()

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 1)
            self.assertEqual(dict(rows[0]), row)
        finally:
            os.unlink(path)

    def test_context_manager(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            path = f.name

        try:
            with TypedCSVWriter(path, Exp1Row) as writer:
                writer.write(_make_row(Exp1Row))

            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
        finally:
            os.unlink(path)

    def test_exp11_shared_by_three_experiments(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            path = f.name

        try:
            writer = TypedCSVWriter(path, Exp11Row)
            for exp_name in ["exp11", "exp12", "exp13"]:
                row = _make_row(Exp11Row)
                row["experiment"] = exp_name
                writer.write(row)
            writer.close()

            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            experiments = {r["experiment"] for r in rows}
            self.assertEqual(experiments, {"exp11", "exp12", "exp13"})
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
