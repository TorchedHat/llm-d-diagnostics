import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from owned_process import check, stop


def test_only_owner_can_stop_the_collector(tmp_path):
    script = tmp_path / "collector.py"
    script.write_text("import time\nwhile True: time.sleep(1)\n")
    record = tmp_path / "owner.json"
    controller = Path(__file__).parent.parent / "owned_process.py"
    parent = subprocess.Popen([sys.executable, str(controller), "start", str(record), "token", str(script)])
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                json.loads(record.read_text())
                break
            except (OSError, ValueError):
                time.sleep(.02)
        assert check(record, "token") == 0
        assert check(record, "other") == 1
        with pytest.raises(RuntimeError, match="ownership"):
            stop(record, "other")
        assert check(record, "token") == 0
        assert stop(record, "token") == 0
        parent.wait(timeout=5)
        assert not record.exists()
    finally:
        if record.exists():
            stop(record, "token")
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=5)


def test_pid_reuse_does_not_signal_an_unrelated_process(tmp_path):
    record = tmp_path / "owner.json"
    record.write_text(json.dumps({"pid": os.getpid(), "start": "not-the-current-start",
                                  "token": "token", "script": "collector.py"}))
    with pytest.raises(RuntimeError, match="PID has been reused"):
        stop(record, "token")
    assert record.exists()
