"""The CI simulator build must match the version the sim manifests deploy."""

import glob
import os
import re
import unittest

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


class TestSimVersion(unittest.TestCase):

    def test_ci_sim_version_matches_manifests(self):
        tags = set()
        for path in glob.glob(os.path.join(REPO, "manifests", "sim", "*.yaml")):
            with open(path) as fh:
                tags |= set(re.findall(r"llm-d-inference-sim:(v[0-9.]+)", fh.read()))
        self.assertEqual(len(tags), 1, f"manifests/sim/ use several sim versions: {tags}")

        with open(os.path.join(REPO, ".github", "workflows", "ci.yml")) as fh:
            ci = fh.read()
        match = re.search(r"SIM_VERSION:\s*(v[0-9.]+)", ci)
        self.assertIsNotNone(match, "SIM_VERSION not set in ci.yml")
        self.assertEqual(match.group(1), tags.pop(),
                         "bump SIM_VERSION and SIM_COMMIT in ci.yml with manifests/sim/")


if __name__ == "__main__":
    unittest.main()
