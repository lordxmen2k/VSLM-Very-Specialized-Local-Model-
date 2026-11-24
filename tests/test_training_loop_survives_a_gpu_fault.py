"""The training loop must survive a GPU fault, and say so.

Run 14 died at step 520 of 1,250:

    GPU fault (AcceleratorError); clearing the cache and retrying (attempt 1)
    ...
    During handling of the above exception, another exception occurred:

    File "src\\la\\train.py", line 755, in main
        win_tok -= int(batch["attention_mask"].sum().item())
    torch.AcceleratorError: CUDA error: out of memory

The retry machinery was right. The fault was recognised, the cache was
cleared, and the batch was going to be run again. Then the recovery
path asked the card for a number, and the card - which had just run out
of memory - said no, this time outside the `try` that was meant to
catch it. A three-and-a-half-hour run died on a bookkeeping line.

The old tests for this were source-text assertions: they checked that
the file *contained* the string `is_gpu_fault(exc)` and the string
`torch.cuda.empty_cache()`. The code contained both, and still died.
What matters is what the recovery path *does*, so these tests run it.

The second half of the defect is pinned down too: the recovery path
read `batch["attention_mask"].sum().item()`, a value living on the
GPU. The fix counts the tokens on the CPU, before the batch moves. The
test below proves the recovery path makes no device reads at all, by
handing it a batch whose device-side read would fail loudly.
"""
import pathlib
import unittest


def _torch_or_skip():
    try:
        import torch
    except ImportError:
        raise unittest.SkipTest("torch is not installed here")
    return torch


class _FakeOptim:
    def __init__(self):
        self.zeroed = 0

    def zero_grad(self, set_to_none=False):
        self.zeroed += 1


class _ExplodingMask:
    """A mask that refuses to be read.

    It stands in for the GPU tensor: reading a value off the device is
    what killed run 14, so the test makes that read impossible and
    checks that the recovery path never attempts it.
    """

    def __init__(self, torch, value):
        self._torch = torch
        self._value = value
        self.reads = 0

    def sum(self):
        self.reads += 1
        raise AssertionError(
            "the recovery path read the batch from the device; on a GPU "
            "that had just run out of memory, this is the read that "
            "raised the fault again and killed run 14")

    def item(self):
        self.reads += 1
        raise AssertionError("the recovery path read the batch from the device")


class TestTheRetryRecoversFromAGpuFault(unittest.TestCase):
    def setUp(self):
        self.torch = _torch_or_skip()
        torch = self.torch
        import la.train as T
        self.T = T

        calls = []

        class FlakyModel:
            def __call__(self, **batch):
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError("CUDA error: out of memory")
                loss = torch.tensor(1.0, requires_grad=True)
                return type("Out", (), {"loss": loss})()

        self.model = FlakyModel()
        self.calls = calls

    def test_a_fault_is_retried_and_the_batch_still_runs(self):
        torch = self.torch
        T = self.T
        optim = _FakeOptim()
        batch = {"input_ids": torch.tensor([[1, 2, 3]])}

        out, ok = T.run_batch_with_retries(self.model, batch, 16, 3, optim)

        self.assertTrue(ok, "the batch was not recovered")
        self.assertIsNotNone(out)
        self.assertEqual(len(self.calls), 2,
                         "the batch was not run again after the fault")

    def test_a_fault_that_never_clears_skips_the_batch(self):
        torch = self.torch
        T = self.T
        optim = _FakeOptim()

        class AlwaysFaults:
            def __call__(self, **batch):
                raise RuntimeError("CUDA error: out of memory")

        out, ok = T.run_batch_with_retries(AlwaysFaults(), {"input_ids":
                                                            torch.tensor([[1]])},
                                           16, 2, optim)
        self.assertFalse(ok, "a batch that never ran was reported as run")
        self.assertIsNone(out)
        self.assertGreaterEqual(optim.zeroed, 1,
                                "the gradients were not cleared after a "
                                "skipped batch")

    def test_an_ordinary_bug_is_not_retried(self):
        """A TypeError is a bug in the code and must fail loudly."""
        torch = self.torch
        T = self.T
        optim = _FakeOptim()

        class Buggy:
            def __call__(self, **batch):
                raise TypeError("unsupported operand type(s)")

        with self.assertRaises(TypeError):
            T.run_batch_with_retries(Buggy(), {"input_ids":
                                                torch.tensor([[1]])},
                                     16, 3, optim)

    def test_the_recovery_path_makes_no_device_read(self):
        """The heart of run 14: the retry must not read the batch."""
        torch = self.torch
        T = self.T
        optim = _FakeOptim()
        mask = _ExplodingMask(torch, 3)
        batch = {"input_ids": torch.tensor([[1, 2, 3]]),
                 "attention_mask": mask}

        out, ok = T.run_batch_with_retries(self.model, batch, 16, 3, optim)

        self.assertTrue(ok, "the batch was not recovered")
        self.assertEqual(mask.reads, 0,
                         "the recovery path read the attention mask, which "
                         "lives on the device")


class TestTheTokenCountIsTakenOnTheCpu(unittest.TestCase):
    def setUp(self):
        self.torch = _torch_or_skip()
        torch = self.torch
        import la.train as T
        self.T = T

        calls = []

        class FlakyModel:
            def __call__(self, **batch):
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError("CUDA error: out of memory")
                loss = torch.tensor(1.0, requires_grad=True)
                return type("Out", (), {"loss": loss})()

        self.model = FlakyModel()
        self.calls = calls

    def test_the_count_survives_a_fault_that_never_reads_the_device(self):
        """The whole sequence, with the device read made impossible."""
        torch = self.torch
        T = self.T
        optim = _FakeOptim()
        mask = _ExplodingMask(torch, 3)
        batch = {"input_ids": torch.tensor([[1, 2, 3]]),
                 "attention_mask": mask}

        # this is what the trainer does: count on the CPU, then move
        batch_tokens = T.count_batch_tokens(
            {"attention_mask": torch.tensor([[1, 1, 1]])})
        self.assertEqual(batch_tokens, 3)

        win_tok = batch_tokens
        moved = {k: v for k, v in batch.items()}

        def correct(_attempt, _n=batch_tokens):
            nonlocal win_tok
            win_tok -= _n

        out, ok = T.run_batch_with_retries(self.model, moved, 16, 3, optim,
                                           on_retry=correct)
        self.assertTrue(ok)
        self.assertEqual(mask.reads, 0,
                         "the device was read during the recovery")
        self.assertEqual(win_tok, 0,
                         "the token counter was not corrected after the retry")


if __name__ == "__main__":
    unittest.main()
