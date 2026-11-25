"""The allocator must not hoard memory for the whole run.

Run 15 trained for 25 minutes and died at step 100:

```
  step    100/1,250  loss 0.0949  ... 221.53 s/step  vram 7.9/22.4 GB
  GPU fault (AcceleratorError); clearing the cache and retrying (attempt 1)
  ...
  the GPU fault did not clear after 3 retries; skipping this batch and carrying on
Traceback (most recent call last):
  File "src\\la\\train.py", line 790, in main
    batch = {k: v.to(model.device) for k, v in batch.items()}
torch.AcceleratorError: CUDA error: out of memory
```

The reserved number is the story: **22.4 GB reserved on a 12.88 GB card.**
The allocated peak was 7.9 GB, so only 5 GB was actually in use. The rest
was PyTorch's caching allocator holding freed blocks so it could hand them
out again quickly.

That is normally a good trade. Here it was fatal. The transfer of the next
batch needs a *fresh* allocation, and a fragmented allocator with 22.4 GB
reserved against a 12.88 GB card cannot serve one — so the run died at
exactly the point where it asked the card for something new.

The step time tells the same story from the other side: 221.53 s/step at
step 100, against 9.13 s/step in run 14 at the same step with the same
corpus and the same cap. That is not more compute. That is memory
pressure, the allocator thrashing to find a block.

**THE FIX: give the memory back.** `PYTORCH_CUDA_ALLOC_CONF` with
`expandable_segments:True` stops the allocator from fragmenting the way
the default does, and an explicit `empty_cache()` between batches returns
freed blocks to the driver instead of hoarding them for three hours.

The number this has to beat: reserved must stay under the card's 12.88 GB,
not sit at 22.4 GB.
"""
import pathlib
import unittest

TRAIN_PY = pathlib.Path(__file__).resolve().parents[1] / "src" / "la" / "train.py"


class TestTheAllocatorIsConfigured(unittest.TestCase):
    """The allocator's configuration must be set before the model loads."""

    def setUp(self):
        self.source = TRAIN_PY.read_text(encoding="utf-8")

    def test_expandable_segments_is_requested(self):
        """expandable_segments:True is what stops the fragmentation."""
        self.assertIn("expandable_segments", self.source,
                      "the allocator is left on its default settings, so it "
                      "fragments and hoards freed blocks for the whole run")

    def test_the_allocator_conf_is_set_before_the_model_is_built(self):
        """It has to be set before the first allocation, or it does nothing.

        The setting is made at import time, before torch is imported and
        long before `main` runs, so the check is that the module-level
        call happens above `import torch` - torch reads the variable
        when it creates its allocator, which is the first CUDA
        allocation.
        """
        source = self.source
        conf = source.find("configure_allocator()")
        torch_import = source.find("import torch")
        build = source.find("build_model(")
        self.assertTrue(conf >= 0, "the allocator is not configured at all")
        self.assertTrue(torch_import >= 0, "torch is not imported")
        self.assertTrue(build >= 0, "the model is not built")
        self.assertLess(conf, torch_import,
                        "the allocator is configured after torch is imported, "
                        "so the setting comes too late to affect the blocks "
                        "torch already reserved")
        self.assertLess(conf, build,
                        "the allocator is configured after the model is "
                        "built, so the setting has no effect on the blocks "
                        "that were already reserved")


class TestTheCacheIsGivenBack(unittest.TestCase):
    """The allocator must not hold the whole run's memory."""

    def setUp(self):
        self.source = TRAIN_PY.read_text(encoding="utf-8")

    def test_the_cache_is_emptied_during_training(self):
        self.assertIn("torch.cuda.empty_cache()", self.source,
                      "the allocator never gives memory back during a run")

    def test_the_cache_is_emptied_in_the_training_loop(self):
        """Not just in the fault handler - the normal path needs it too."""
        import ast
        tree = ast.parse(self.source)
        main = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        body = ast.unparse(main)
        self.assertIn("empty_cache", body,
                      "the training loop never empties the cache, so the "
                      "allocator hoards every freed block for the whole run")


class TestTheSpeedTestFlagsFragmentation(unittest.TestCase):
    """The speed test must call out a fragmented reservation.

    The speed test used to say that a reserved figure over the card "is
    not a failure by itself", and run 15's 22.4 GB reservation was read
    through that sentence and dismissed. A reservation much larger than
    the card is the allocator holding blocks it cannot reuse, and it
    grows until the run cannot serve a fresh allocation.
    """

    def setUp(self):
        self.source = TRAIN_PY.read_text(encoding="utf-8")

    def test_a_large_reservation_is_flagged(self):
        self.assertIn("FRAGMENTED", self.source,
                      "the speed test does not flag a fragmented reservation, "
                      "so the number that killed run 15 is not called out")

    def test_the_dismissal_is_gone(self):
        self.assertNotIn("it is not a failure", self.source,
                         "the speed test still says a reservation over the "
                         "card is not a failure, which is what let run 15's "
                         "22.4 GB reservation through")

    def test_the_flag_is_about_the_reservation_not_the_allocation(self):
        """A big reserved-minus-allocated gap is the signal."""
        self.assertIn("card * 1.5", self.source,
                      "the fragmentation check does not compare the "
                      "reservation against the card")


if __name__ == "__main__":
    unittest.main()
