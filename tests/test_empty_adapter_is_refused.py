"""An adapter directory that holds no adapter must stop the benchmark.

The trainer creates the output directory before the training loop and
writes the adapter into it at the very end. A run that crashed - and run
11 crashed, with a cuBLAS failure at the first optimiser step - left an
EMPTY directory behind.

Scoring that directory fell through to the base model. The benchmark
produced a full 660-problem score and reported it as the fine-tune's,
and the number is indistinguishable from a real one. The tokenizer load
failed first, with an error about a missing backend tokenizer, which
reads like a dependency problem rather than a missing training run.

So there are two guards here, and they have to fire before any number is
produced:

  1. the tokenizer comes from the BASE, so the scoring path does not
     depend on the adapter directory existing;
  2. the benchmark refuses an adapter directory with no adapter in it.
"""
import os
import pathlib
import shutil
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestEmptyAdapterIsRefused(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vslm-adapter-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_an_empty_directory_is_not_an_adapter(self):
        from la.bench import _adapter_has_weights
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        self.assertFalse(_adapter_has_weights(empty),
                         "an empty directory is not an adapter")

    def test_a_directory_with_the_marker_is_not_an_adapter(self):
        from la.bench import _adapter_has_weights
        marked = os.path.join(self.tmp, "marked")
        os.makedirs(marked)
        with open(os.path.join(marked, ".incomplete"), "w", encoding="utf-8") as fh:
            fh.write("training started\n")
        self.assertFalse(_adapter_has_weights(marked),
                         "a run that did not finish is not an adapter")

    def test_a_real_adapter_is_an_adapter(self):
        from la.bench import _adapter_has_weights
        real = os.path.join(self.tmp, "real")
        os.makedirs(real)
        with open(os.path.join(real, "adapter_config.json"), "w",
                  encoding="utf-8") as fh:
            fh.write('{"base_model_name_or_path": "Qwen/Qwen3-1.7B"}')
        self.assertTrue(_adapter_has_weights(real))

    def test_the_benchmark_refuses_an_empty_adapter(self):
        from la.bench import _check_adapter_base
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        with self.assertRaises(SystemExit) as caught:
            _check_adapter_base("Qwen/Qwen3-1.7B", empty)
        self.assertIn("REFUSING TO RUN", str(caught.exception))

    def test_the_benchmark_accepts_a_real_adapter(self):
        from la.bench import _check_adapter_base
        real = os.path.join(self.tmp, "real")
        os.makedirs(real)
        with open(os.path.join(real, "adapter_config.json"), "w",
                  encoding="utf-8") as fh:
            fh.write('{"base_model_name_or_path": "Qwen/Qwen3-1.7B"}')
        # the same base, so the guard passes
        _check_adapter_base("Qwen/Qwen3-1.7B", real)

    def test_a_missing_directory_is_not_an_adapter_either(self):
        from la.bench import _adapter_has_weights
        self.assertFalse(_adapter_has_weights(os.path.join(self.tmp, "nope")))


class TestTheTrainerMarksAnUnfinishedRun(unittest.TestCase):
    """The trainer writes a marker and removes it when the run succeeds."""

    def test_the_marker_is_written_and_removed(self):
        source = (ROOT / "src" / "la" / "train.py").read_text(encoding="utf-8")
        self.assertIn(".incomplete", source,
                      "the trainer does not mark an unfinished run")
        # the marker is written before the training loop and removed after it
        written = source.index('open(_marker, "w"')
        loop = source.index("for epoch in range")
        removed = source.index("os.remove(_marker)")
        self.assertLess(written, loop,
                        "the marker must be written before the training loop")
        self.assertGreater(removed, loop,
                           "the marker must be removed after the training loop")
        # and the removal is guarded, so a failed removal does not stop the
        # run that has already saved the adapter
        self.assertIn("except OSError", source[removed - 200:removed + 200])


if __name__ == "__main__":
    unittest.main()
