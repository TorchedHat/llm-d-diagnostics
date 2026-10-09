"""run.sh must forward every parameter the toolkit reads to the in-pod run.

Experiments run inside a pod via `oc exec ... env $REMOTE_ENV`; a variable
missing from REMOTE_ENV is silently replaced by the script's default.
"""

import glob
import os
import re
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


if __name__ == "__main__":
    unittest.main()
