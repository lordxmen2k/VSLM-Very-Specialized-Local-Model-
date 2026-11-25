"""A fault while moving a batch to the device must not kill the run.

Run 15 trained for 25 minutes and died at step 100:

```
  step    100/1,250  loss 0.0949  ... 221.53 s/step  vram 7.9/22.4 GB
  GPU fault (AcceleratorError); clearing the cache and retrying (attempt 1)
  GPU fault (AcceleratorError); clearing the cache and retrying (attempt 2)
  GPU fault (AcceleratorError); clearing the cache and retrying (attempt 3)
  the GPU fault did not clear after 3 retries; skipping this batch and carrying on
Traceback (most recent call last):
  File "src\\la\\train.py", line 790, in main
    batch = {k: v.to(model.device) for k, v in batch.items()}
torch.AcceleratorError: CUDA error: out of memory
```

**The run-15 fix worked.** The recovery path survived the fault this time:
it retried three times, skipped the batch, and carried on — where run 14 died
at the first fault because the recovery path asked the GPU for a number.

But the run then died at the NEXT place the GPU is asked for something: the
transfer of the following batch onto the device. That transfer sits outside
the retry entirely, so an out-of-memory fault there is fatal.

The same rule applies: **the GPU is in an unknown state, so the code must
survive it.** A batch that cannot be moved to the device is a batch to skip,
not a run to abandon — the run is three and a half hours and there are 19,999
other batches.

The memory arithmetic explains why this is worth handling. The logits are the
dominant term: one example of L tokens against a vocabulary of 151,936 costs

    L * 151936 * 4 bytes      (the logits)
    L * 151936 * 4 bytes      (their gradient)

which at the 2,560-token cap is 3.11 GB in fp32, for ONE example of batch
size 1. A 12.88 GB card has no room for a transient spike on top of that.
"""
import pathlib
import unittest

TRAIN_PY = pathlib.Path(__file__).resolve().parents[1] / "src" / "la" / "train.py"


def _torch_or_skip():
    try:
        import torch
    except ImportError:
        raise unittest.SkipTest("torch is not installed here")
    return torch


def _oom(torch):
    """An out-of-memory error, whatever this build calls it.

    `torch.cuda.OutOfMemoryError` cannot be constructed on a CPU-only
    build, so fall back to the RuntimeError that torch raises for the
    same condition.
    """
    cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if cls is not None:
        try:
            return cls("CUDA out of memory. Tried to allocate 2.00 GiB")
        except Exception:
            pass
    return RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


class _FakeOptim:
    def __init__(self):
        self.zeroed = 0

    def zero_grad(self, set_to_none=False):
        self.zeroed += 1


class TestTheTransferIsFaultTolerant(unittest.TestCase):
    """Moving a batch to the device may fail, and that is survivable."""

    def setUp(self):
        self.torch = _torch_or_skip()
        import la.train as T
        self.T = T

    def test_a_fault_while_moving_the_batch_is_caught(self):
        """The transfer is inside the fault handling, not outside it."""
        torch = self.torch
        T = self.T
        optim = _FakeOptim()

        moved = []

        class ExplodingTensor:
            """A tensor whose transfer fails the way a real OOM does."""

            def to(self, device):
                raise _oom(torch)

        batch = {"input_ids": ExplodingTensor(),
                 "attention_mask": torch.tensor([[1, 1, 1]])}

        out, ok = T.move_batch_to_device(
            batch, "cpu", optim, gpu_retries=3,
            on_move=lambda attempt: moved.append(attempt))

        self.assertFalse(ok, "a batch that could not be moved was reported "
                             "as moved")
        self.assertIsNone(out)
        self.assertGreaterEqual(optim.zeroed, 1,
                                "the gradients were not cleared after a "
                                "skipped batch")

    def test_a_fault_while_moving_the_batch_is_retried(self):
        """The transfer is retried, like the forward pass."""
        torch = self.torch
        T = self.T
        optim = _FakeOptim()
        attempts = []

        class FlakyTensor:
            def __init__(self):
                self.n = 0

            def to(self, device):
                self.n += 1
                attempts.append(self.n)
                if self.n == 1:
                    raise _oom(torch)
                return torch.tensor([[1, 2, 3]])

        batch = {"input_ids": FlakyTensor(),
                 "attention_mask": torch.tensor([[1, 1, 1]])}

        out, ok = T.move_batch_to_device(batch, "cpu", optim, gpu_retries=3)

        self.assertTrue(ok, "the batch was not recovered")
        self.assertIsNotNone(out)
        self.assertEqual(len(attempts), 2,
                         "the transfer was not retried after the fault")

    def test_an_ordinary_bug_in_the_transfer_is_not_retried(self):
        """A TypeError is a bug in the code and must fail loudly."""
        torch = self.torch
        T = self.T
        optim = _FakeOptim()

        class BuggyTensor:
            def to(self, device):
                raise TypeError("unsupported operand type(s)")

        with self.assertRaises(TypeError):
            T.move_batch_to_device({"input_ids": BuggyTensor()},
                                   "cpu", optim, gpu_retries=3)

    def test_the_real_batch_is_returned_when_the_transfer_succeeds(self):
        torch = self.torch
        T = self.T
        optim = _FakeOptim()
        batch = {"input_ids": torch.tensor([[1, 2, 3]]),
                 "attention_mask": torch.tensor([[1, 1, 1]])}

        out, ok = T.move_batch_to_device(batch, "cpu", optim, gpu_retries=3)

        self.assertTrue(ok)
        self.assertIsNotNone(out)
        self.assertIn("input_ids", out)
        self.assertIn("attention_mask", out)


class TestTheTrainingLoopUsesIt(unittest.TestCase):
    """The loop must call it, and must not transfer the batch itself."""

    def setUp(self):
        import ast
        self.tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))
        self.main = next(n for n in self.tree.body
                         if isinstance(n, ast.FunctionDef) and n.name == "main")
        self.body = ast.unparse(self.main)

    def test_the_loop_calls_move_batch_to_device(self):
        self.assertIn("move_batch_to_device", self.body,
                      "the training loop still transfers the batch itself, "
                      "outside the fault handling")

    def test_the_loop_does_not_transfer_the_batch_inline(self):
        """A bare `.to(model.device)` in the loop is the unfaultable path."""
        self.assertNotIn(".to(model.device)", self.body,
                         "the training loop transfers the batch directly, "
                         "where an out-of-memory fault is fatal")


if __name__ == "__main__":
    unittest.main()
