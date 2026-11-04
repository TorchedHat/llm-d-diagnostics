#!/usr/bin/env python3
"""Start and stop one explicitly identified collector in the current pod."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return {"state": stat[0], "start": stat[19]}
    except (OSError, IndexError):
        return None


def start(record_path, token, script):
    path = Path(record_path)
    # An exclusive owner file prevents two collectors writing the same result directory.
    with path.open("x") as owner:
        env = {**os.environ, "LLMD_COLLECTOR_TOKEN": token}
        process = subprocess.Popen([sys.executable, script], env=env)
        identity = process_identity(process.pid)
        if not identity:
            process.wait()
            raise RuntimeError("collector exited before its identity could be recorded")
        json.dump({"pid": process.pid, "start": identity["start"], "token": token,
                   "script": script}, owner)
        owner.flush()
    return process.wait()


def stop(record_path, token):
    path = Path(record_path)
    record = json.loads(path.read_text())
    if record.get("token") != token or type(record.get("pid")) is not int:
        raise RuntimeError("collector ownership does not match")
    pid = record["pid"]
    identity = process_identity(pid)
    if identity and identity["state"] != "Z":
        if identity["start"] != record.get("start"):
            raise RuntimeError("collector PID has been reused")
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        if (os.fsencode(record["script"]) not in command or
                f"LLMD_COLLECTOR_TOKEN={token}".encode() not in environment):
            raise RuntimeError("collector command or token does not match")
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            current = process_identity(pid)
            if not current or current["state"] == "Z" or current["start"] != identity["start"]:
                break
            time.sleep(.1)
        else:
            raise RuntimeError("collector did not stop; artifacts were not copied")
    path.unlink()
    return 0


def check(record_path, token):
    record = json.loads(Path(record_path).read_text())
    if record.get("token") != token or type(record.get("pid")) is not int:
        return 1
    identity = process_identity(record["pid"])
    return 0 if identity and identity["state"] != "Z" and identity["start"] == record.get("start") else 1


def main():
    try:
        if len(sys.argv) == 5 and sys.argv[1] == "start":
            return start(*sys.argv[2:])
        if len(sys.argv) == 4 and sys.argv[1] == "stop":
            return stop(*sys.argv[2:])
        if len(sys.argv) == 4 and sys.argv[1] == "check":
            return check(*sys.argv[2:])
        raise ValueError("usage: owned_process.py start RECORD TOKEN SCRIPT | stop/check RECORD TOKEN")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"collector control failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
