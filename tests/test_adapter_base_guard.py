"""The benchmark must refuse to score an adapter against the wrong base.

A LoRA adapter is a difference between two models. Applied to the wrong
weights it produces a model that is neither one, and the result is
silently incoherent rather than an error.

Run 5 was trained on Qwen/Qwen3-1.7B (the instruct checkpoint) and the
Makefile scored it against Qwen/Qwen3-1.7B-Base, because the two names
were hardcoded in different places. The merged model was incoherent, it
never emitted its stop token, and every problem ran to the 2048-token cap:
about 35 minutes for a batch of 8, and 48 hours for the benchmark.

These tests pin the guard.
"""
import json
import os
import shutil
import tempfile
import unittest

from la.bench import _check_adapter_base, _same_checkpoint


class TestSameCheckpoint(unittest.TestCase):
    def test_identical_names(self):
        self.assertTrue(_same_checkpoint("Qwen/Qwen3-1.7B",
                                         "Qwen/Qwen3-1.7B"))

    def test_a_local_path_matches_the_hub_id(self):
        self.assertTrue(_same_checkpoint("Qwen/Qwen3-1.7B",
                                         "G:/models/huggingface/hub/"
                                         "models--Qwen--Qwen3-1.7B/snapshot"))

    def test_base_is_not_the_instruct_release(self):
        self.assertFalse(_same_checkpoint("Qwen/Qwen3-1.7B-Base",
                                          "Qwen/Qwen3-1.7B"))

    def test_trailing_slashes_do_not_matter(self):
        self.assertTrue(_same_checkpoint("Qwen/Qwen3-1.7B/",
                                         "Qwen/Qwen3-1.7B"))


class TestCheckAdapterBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, name, payload):
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    def test_a_matching_base_is_allowed(self):
        self.write("adapter_config.json",
                   {"base_model_name_or_path": "Qwen/Qwen3-1.7B"})
        _check_adapter_base("Qwen/Qwen3-1.7B", self.dir)   # must not raise

    def test_a_different_base_is_refused(self):
        self.write("adapter_config.json",
                   {"base_model_name_or_path": "Qwen/Qwen3-1.7B"})
        with self.assertRaises(SystemExit) as cm:
            _check_adapter_base("Qwen/Qwen3-1.7B-Base", self.dir)
        msg = str(cm.exception)
        self.assertIn("REFUSING TO RUN", msg)
        self.assertIn("Qwen/Qwen3-1.7B-Base", msg)
        self.assertIn("Qwen/Qwen3-1.7B", msg)

    def test_train_args_is_used_when_adapter_config_is_absent(self):
        self.write("train_args.json", {"base": "Qwen/Qwen3-1.7B"})
        with self.assertRaises(SystemExit):
            _check_adapter_base("Qwen/Qwen3-1.7B-Base", self.dir)

    def test_an_adapter_with_no_record_is_allowed(self):
        """An adapter from an older run may not carry the record."""
        _check_adapter_base("Qwen/Qwen3-1.7B-Base", self.dir)

    def test_a_missing_adapter_is_allowed(self):
        _check_adapter_base("Qwen/Qwen3-1.7B", os.path.join(self.dir, "nope"))


class TestTheMakefileAndTheConfigAgree(unittest.TestCase):
    """The bug was two hardcoded copies of the base model name."""

    def _repo(self):
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(here)

    def test_the_makefile_reads_the_base_from_the_config(self):
        repo = self._repo()
        with open(os.path.join(repo, "Makefile"), encoding="utf-8") as fh:
            mk = fh.read()
        self.assertNotIn("BASE ?= Qwen/Qwen3-1.7B-Base", mk,
                         "the Makefile must not hardcode the base model")
        self.assertIn("configs/default.yaml", mk,
                      "the Makefile must read the base model from the config")

    def test_the_config_names_the_instruct_checkpoint(self):
        repo = self._repo()
        import yaml
        cfg = yaml.safe_load(open(os.path.join(repo, "configs", "default.yaml"),
                                   encoding="utf-8"))
        self.assertEqual(cfg["model"]["base"], "Qwen/Qwen3-1.7B")

    def test_the_makefile_default_equals_the_config(self):
        """Run make -n and read the base it would use."""
        import subprocess
        repo = self._repo()
        out = subprocess.run(["make", "-n", "bench"], cwd=repo,
                             capture_output=True, text=True)
        if out.returncode != 0 or "--base" not in out.stdout:
            self.skipTest("make is not available here")
        line = [l for l in out.stdout.splitlines() if "la.bench" in l][0]
        self.assertIn("--base Qwen/Qwen3-1.7B ", line)
        self.assertNotIn("--base Qwen/Qwen3-1.7B-Base", line)


if __name__ == "__main__":
    unittest.main()
