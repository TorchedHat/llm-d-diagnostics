"""
llm-d Diagnostics Toolkit — Row schemas, config enums, and typed CSV writing.

Single source of truth for every experiment's CSV schema. Each experiment
gets a TypedDict (row shape) and str Enums (categorical field values).

TypedCSVWriter derives FIELDS from the TypedDict at construction time and
validates every row at write time via ValueError.  Misspelled field names
become immediate errors instead of silent data corruption.

Zero external dependencies.  Python 3.9+.
"""

from __future__ import annotations

import csv
import threading
from enum import Enum
from typing import TypedDict, get_type_hints

# ── Helper ──────────────────────────────────────────────────────────────────

def fields_for(row_type: type) -> list[str]:
    """Extract ordered field names from a TypedDict class."""
    return list(get_type_hints(row_type).keys())


# ── Config / phase enums ────────────────────────────────────────────────────
#
# All inherit from str so they serialize to CSV without conversion.
# Per-experiment enums keep semantic domains separate.
#
# Naming convention note:
#   ConfigExp1 uses "DISAGG-D1" / "DISAGG-D2" — identifies WHICH decode pod
#   (pod identity matters for single-request routing).
#   ConfigThroughput uses "DISAGG-1D" / "DISAGG-2D" — identifies HOW MANY
#   decode pods (topology matters for throughput/saturation measurement).
#   These are intentionally different; do not "fix" the naming.

# exp1: single-request latency
class ConfigExp1(str, Enum):
    BASELINE = "BASELINE"
    DISAGG_D1 = "DISAGG-D1"
    DISAGG_D2 = "DISAGG-D2"
    DISAGG_EPP = "DISAGG-EPP"

# exp1b / exp5: decomposition and sequence-length sweep
class ConfigDecompose(str, Enum):
    A_PREFILL_DIRECT = "A-prefill-direct"
    B_DECODE_DIRECT = "B-decode-direct"
    C_SIDECAR_ONLY = "C-sidecar-only"
    D_DISAGGREGATED = "D-disaggregated"
    E_EPP_GATEWAY = "E-EPP-gateway"

# exp2 / exp6: throughput and saturation
class ConfigThroughput(str, Enum):
    BASELINE = "BASELINE"
    DISAGG_1D = "DISAGG-1D"
    DISAGG_2D = "DISAGG-2D"
    DISAGG_EPP = "DISAGG-EPP"

# exp3 / exp7: isolation and mixed workload (same config space)
class ConfigPaired(str, Enum):
    """BASELINE vs DISAGG-2D comparison. Used by exp3 (isolation) and exp7 (mixed)."""
    BASELINE = "BASELINE"
    DISAGG_2D = "DISAGG-2D"
    DISAGG_EPP = "DISAGG-EPP"

# Aliases for backward compatibility and import clarity
ConfigIsolation = ConfigPaired
ConfigMixed = ConfigPaired

# exp3 / exp7: request weight
class Weight(str, Enum):
    HEAVY = "heavy"
    LIGHT = "light"

# exp7: workload class and priority
class WorkloadClass(str, Enum):
    SHORT = "short"
    LONG = "long"

class Priority(str, Enum):
    HIGH = "high"
    LOW = "low"


def priority_for(wl: WorkloadClass) -> Priority:
    """Priority is a deterministic function of workload class, not independent data."""
    return Priority.LOW if wl == WorkloadClass.LONG else Priority.HIGH

# exp8: prefix cache phases and states
class CachePhase(str, Enum):
    HIT_MISS = "hit_miss"
    MULTI_TURN = "multi_turn"
    DECAY = "decay"
    EPP_ROUTING = "epp_routing"

class CacheState(str, Enum):
    COLD = "cold"
    WARM_SAME_POD = "warm_same_pod"
    WARM_DIFF_POD = "warm_diff_pod"
    CONTROL_DIFF_PROMPT = "control_diff_prompt"
    # Partial enumeration: multi-turn phase uses dynamic "turn_1".."turn_N"
    # strings that bypass this enum.  The CSV column type is str in all cases;
    # this enum covers hit_miss and decay phases only.
    DECAY_PRIME = "decay_prime"
    DECAY_WARM = "decay_warm"
    DECAY_AFTER_WAIT = "decay_after_wait"
    EPP_FIRST = "epp_first"
    EPP_REPEAT = "epp_repeat"

# exp9: model load phases
class ModelLoadPhase(str, Enum):
    POD_DELETE = "pod_delete"
    WEIGHT_LOAD = "weight_load"
    NIXL_INIT = "nixl_init"
    COMPILE = "compile"
    TOTAL_STARTUP = "total_startup"
    WALL_TOTAL = "wall_total"

# exp10: KV eviction phases
class EvictionPhase(str, Enum):
    DIST_COLD = "dist_cold"
    DIST_WARM = "dist_warm"
    WARM = "warm"
    AFTER_DELAY = "after_delay"
    BG_WARM = "bg_warm"
    BG_AFTER_DELAY = "bg_after_delay"
    PRESSURE_WARM = "pressure_warm"
    PRESSURE_AFTER = "pressure_after"

# exp10: cache hit indicator
class CacheHit(str, Enum):
    YES = "yes"
    NO = "no"
    NA = "n/a"


# ── Row TypedDicts ──────────────────────────────────────────────────────────
#
# All values are str because CSV is text.  The TypedDict enforces key
# completeness — mypy catches missing or extra keys.  Field order matches
# the CSV column order (TypedDict preserves insertion order in 3.7+).

class Exp1Row(TypedDict):
    experiment: str
    config: str
    run: str
    prompt_tokens_target: str
    max_tokens: str
    ttft_ms: str
    total_ms: str
    status_code: str
    prompt_tokens_actual: str
    completion_tokens: str
    pod: str
    error: str


class Exp1bRow(TypedDict):
    experiment: str
    config: str
    run: str
    pod: str
    ttft_ms: str
    total_ms: str
    status_code: str
    completion_tokens: str
    error: str


class Exp2Row(TypedDict):
    experiment: str
    config: str
    run: str
    concurrency: str
    pod: str
    ttft_ms: str
    total_ms: str
    status_code: str
    target: str
    error: str


class Exp3Row(TypedDict):
    experiment: str
    config: str
    trial: str
    weight: str
    idx: str
    pod: str
    ttft_ms: str
    total_ms: str
    status_code: str
    completion_tokens: str
    itl_ms: str
    error: str


class Exp4Row(TypedDict):
    experiment: str
    sub: str
    phase: str
    timestamp: str
    epoch_ms: str
    ttft_ms: str
    total_ms: str
    itl_mean_ms: str
    itl_p99_ms: str
    token_count: str
    status_code: str
    note: str
    error: str
    detect_epoch_ms: str
    recover_epoch_ms: str
    probes_to_detect: str
    probes_to_recover: str


class Exp5Row(TypedDict):
    experiment: str
    config: str
    pod: str
    prompt_tokens_target: str
    run: str
    ttft_ms: str
    total_ms: str
    status_code: str
    prompt_tokens_actual: str
    completion_tokens: str
    error: str


class Exp5bRow(TypedDict):
    experiment: str
    pod: str
    prompt_tokens_target: str
    prompt_tokens_actual: str
    run: str
    ttft_ms: str
    total_ms: str
    status_code: str
    error: str
    nixl_bytes_delta: str
    nixl_xfer_time_delta_ms: str
    nixl_transfers_delta: str


class Exp6Row(TypedDict):
    experiment: str
    config: str
    qps_target: str
    seq: str
    depart_delay_ms: str
    ttft_ms: str
    total_ms: str
    status_code: str
    completion_tokens: str
    error: str
    pod: str


class Exp7Row(TypedDict):
    experiment: str
    config: str
    seq: str
    workload_class: str
    priority: str
    prompt_tokens_target: str
    max_tokens: str
    pod: str
    ttft_ms: str
    total_ms: str
    itl_mean_ms: str
    itl_p99_ms: str
    status_code: str
    completion_tokens: str
    scheduled_at_s: str
    depart_delay_ms: str
    error: str


class Exp7GpuRow(TypedDict):
    experiment: str
    config: str
    sample_time_s: str
    gpu_index: str
    gpu_util_pct: str
    mem_util_pct: str


class Exp8Row(TypedDict):
    experiment: str
    phase: str
    config: str
    run: str
    pod: str
    prompt_tokens: str
    turn: str
    cache_state: str
    trial_order: str
    ttft_ms: str
    total_ms: str
    status_code: str
    completion_tokens: str
    error: str


class Exp9Row(TypedDict):
    experiment: str
    run: str
    pod_deleted: str
    pod_new: str
    phase: str
    duration_ms: str
    wall_total_ms: str
    timestamp: str
    log_line: str


class Exp10Row(TypedDict):
    experiment: str
    pod: str
    run: str
    delay_s: str
    background_load: str
    phase: str
    ttft_ms: str
    total_ms: str
    status_code: str
    prompt_tokens: str
    completion_tokens: str
    cache_hit: str
    error: str


class Exp11Row(TypedDict):
    """Throughput scaling (exp11/exp12/exp13). Shared schema."""
    experiment: str
    config: str
    prompt_tokens_target: str
    max_tokens: str
    concurrency: str
    run: str
    ttft_ms: str
    total_ms: str
    status_code: str
    prompt_tokens_actual: str
    completion_tokens: str
    target: str
    error: str


class Exp14Row(TypedDict):
    """Overhead decomposition under concurrent load."""
    experiment: str
    config: str
    pod: str
    prompt_tokens_target: str
    concurrency: str
    run: str
    ttft_ms: str
    total_ms: str
    status_code: str
    prompt_tokens_actual: str
    completion_tokens: str
    error: str


# ── Typed CSV Writer ────────────────────────────────────────────────────────

class TypedCSVWriter:
    """Thread-safe CSV writer parameterized on a TypedDict row type.

    Derives field names from the TypedDict at construction time.
    Validates every row's keys match the schema (raises ValueError on mismatch).

    Usage:
        writer = TypedCSVWriter("out.csv", Exp1Row)
        writer.write({"experiment": "exp1", "config": "BASELINE", ...})
        writer.close()
    """

    def __init__(self, filepath: str, row_type: type):
        self.filepath = filepath
        self.row_type = row_type
        self.fieldnames = fields_for(row_type)
        self._lock = threading.Lock()
        self._file = open(filepath, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def write(self, row: dict) -> None:
        row_keys = set(row.keys())
        expected = set(self.fieldnames)
        if row_keys != expected:
            missing = expected - row_keys
            extra = row_keys - expected
            parts = []
            if missing:
                parts.append(f"missing={missing}")
            if extra:
                parts.append(f"extra={extra}")
            raise ValueError(
                f"{self.row_type.__name__} schema mismatch: {', '.join(parts)}"
            )
        with self._lock:
            self._writer.writerow(row)
            self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
