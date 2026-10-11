"""run.sh must forward every parameter the toolkit reads to the in-pod run.

Experiments run inside a pod via `oc exec ... env $REMOTE_ENV`; a variable
missing from REMOTE_ENV is silently replaced by the script's default.
"""

import glob
import os
import re
import subprocess
import unittest

TOOLKIT = os.path.join(os.path.dirname(__file__), "..")


def _read(path):
    with open(path) as fh:
        return fh.read()


class TestRunShForwarding(unittest.TestCase):

    def test_every_env_parameter_is_forwarded(self):
        used = set()
        for path in glob.glob(os.path.join(TOOLKIT, "*.py")):
            used |= set(re.findall(r'env\("([A-Z_0-9]+)"', _read(path)))

        run_sh = _read(os.path.join(TOOLKIT, "run.sh"))
        block = re.search(r'EXPERIMENT_ENV_VARS="(.*?)"', run_sh, re.S)
        self.assertIsNotNone(block, "EXPERIMENT_ENV_VARS not found in run.sh")
        forwarded = set(block.group(1).split())
        for line in re.findall(r'^REMOTE_ENV="(.*)"$', run_sh, re.M):
            forwarded |= set(re.findall(r"([A-Z_0-9]+)=", line))

        missing = sorted(used - forwarded)
        self.assertEqual(missing, [], f"add to EXPERIMENT_ENV_VARS in run.sh: {missing}")


class TestPreflightEndpoints(unittest.TestCase):
    """needed_endpoints, extracted from run.sh and run under bash."""

    def _needed(self, script, **env):
        run_sh = _read(os.path.join(TOOLKIT, "run.sh"))
        func = re.search(r"^needed_endpoints\(\) \{.*?^\}$", run_sh, re.S | re.M).group(0)
        base = {"BASELINE_URL": "http://b", "DISAGG_D1_URL": "http://d1",
                "DISAGG_D2_URL": "http://d2", "EPP_URL": "", "PATH": os.environ["PATH"]}
        out = subprocess.run(["bash", "-c", func + '\nneeded_endpoints "$1"', "_", script],
                             env=base | env, capture_output=True, text=True, check=True)
        return [line.split("|")[0] for line in out.stdout.split()]

    def test_calibrate_needs_one_instance(self):
        self.assertEqual(self._needed("exp18_calibrate.py"), ["calibrate"])

    def test_configs_limit_the_probes(self):
        self.assertEqual(self._needed("exp16_per_token_trace.py", CONFIGS="BASELINE"), ["baseline"])
        self.assertEqual(self._needed("exp12_tput_outlen.py", CONFIGS="DISAGG-2D"),
                         ["manual-decode", "manual-decode-2"])

    def test_default_probes_everything_configured(self):
        self.assertEqual(self._needed("exp1_latency.py", EPP_URL="http://e"),
                         ["baseline", "manual-decode", "manual-decode-2", "epp"])
        self.assertEqual(self._needed("exp1_latency.py"),
                         ["baseline", "manual-decode", "manual-decode-2"])


if __name__ == "__main__":
    unittest.main()
