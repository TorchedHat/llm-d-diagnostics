"""Offline checks for the deployment script's rendered manifests."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy.sh"


class DeployScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.cluster_dir = self.root / "cluster"
        self.cluster_dir.mkdir()
        (self.cluster_dir / "env.sh").write_text(
            'export NS="issue-12-test"\n'
            'export MODEL="test-model"\n'
            'export GPU_RESOURCE_NAME="amd.com/gpu"\n'
            'export SIDECAR_SCHEME="http"\n'
        )

        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        fake_oc = bin_dir / "oc"
        fake_oc.write_text(
            "#!/bin/sh\n"
            "set -eu\n"
            "printf '%s\\n' \"$*\" >> \"$OC_CALL_LOG\"\n"
            "read_file=0\n"
            "for arg in \"$@\"; do\n"
            "  if [ \"$read_file\" -eq 1 ]; then\n"
            "    if [ \"$arg\" = \"-\" ]; then\n"
            "      cat >> \"$OC_MANIFEST_LOG\"\n"
            "    else\n"
            "      cat \"$arg\" >> \"$OC_MANIFEST_LOG\"\n"
            "    fi\n"
            "    printf '\\n---\\n' >> \"$OC_MANIFEST_LOG\"\n"
            "    read_file=0\n"
            "  elif [ \"$arg\" = \"-f\" ]; then\n"
            "    read_file=1\n"
            "  fi\n"
            "done\n"
        )
        fake_oc.chmod(0o755)
        self.calls_file = self.root / "oc-calls.txt"
        self.manifests_file = self.root / "rendered.yaml"

    def run_deploy(self, mode="gpu", **overrides):
        env = os.environ.copy()
        env.update({
            "PATH": f"{self.root / 'bin'}:{env.get('PATH', '')}",
            "DRY_RUN": "client",
            "OC_CALL_LOG": str(self.calls_file),
            "OC_MANIFEST_LOG": str(self.manifests_file),
        })
        env.update(overrides)
        args = [str(DEPLOY_SCRIPT), str(self.cluster_dir)]
        if mode:
            args.append(mode)
        return subprocess.run(args, env=env, capture_output=True, text=True)

    def read_calls(self):
        return self.calls_file.read_text() if self.calls_file.exists() else ""

    def read_manifests(self):
        return self.manifests_file.read_text() if self.manifests_file.exists() else ""

    def test_gpu_dry_run_uses_current_llmd_images_flags_and_gpu_resource(self):
        result = self.run_deploy()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

        rendered = self.read_manifests()
        self.assertIn("image: vllm/vllm-openai:v0.26.0", rendered)
        self.assertIn(
            "image: ghcr.io/llm-d/llm-d-router-disagg-sidecar:v0.10.0",
            rendered,
        )
        self.assertIn("--kv-connector=nixlv2", rendered)
        self.assertIn("--model-server-port=8001", rendered)
        self.assertIn('--secure-proxy=false', rendered)
        self.assertIn('amd.com/gpu: "1"', rendered)
        self.assertNotIn("nvidia.com/gpu", rendered)
        self.assertNotIn("--connector=nixlv2", rendered)
        self.assertIn("--dry-run=client", self.read_calls())
        self.assertNotIn("delete", self.read_calls())

    def test_sim_dry_run_applies_sim_manifests_without_gpu_resources(self):
        result = self.run_deploy("sim")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

        rendered = self.read_manifests()
        calls = self.read_calls()
        self.assertIn("kind: ConfigMap", rendered)
        self.assertIn("MODEL_NAME: \"test-model\"", rendered)
        self.assertIn(
            "image: ghcr.io/llm-d/llm-d-inference-sim:v0.10.2",
            rendered,
        )
        self.assertIn(
            "image: ghcr.io/llm-d/llm-d-router-disagg-sidecar:v0.10.0",
            rendered,
        )
        self.assertIn("--model-server-port=8001", rendered)
        self.assertIn("name: test-client", rendered)
        self.assertNotIn("nvidia.com/gpu", rendered)
        self.assertNotIn("vllm/vllm-openai", rendered)
        self.assertIn("manifests/sim/20-sim-decode.yaml", calls)
        self.assertIn("manifests/sim/22-sim-decode-2.yaml", calls)
        self.assertNotIn("rollout", calls)
        self.assertNotIn("delete", calls)

    def test_rejects_invalid_gpu_resource_before_applying(self):
        (self.cluster_dir / "env.sh").write_text(
            'export NS="issue-12-test"\n'
            'export MODEL="test-model"\n'
            'export GPU_RESOURCE_NAME="nvidia.com/gpu:bad"\n'
        )
        result = self.run_deploy()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid GPU_RESOURCE_NAME", result.stderr + result.stdout)
        self.assertEqual(self.read_calls(), "")


if __name__ == "__main__":
    unittest.main()
