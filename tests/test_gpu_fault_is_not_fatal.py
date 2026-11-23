"""A GPU fault must not kill a three-and-a-half-hour training run.

Run 11 died at the first optimiser step:

    RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling
    `cublasSgemm(handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb,
    &beta, c, ldc)`

The corpus had been generated, the contract checks had passed and the
model had loaded. Everything before the run was fine, and the run itself
was three and a half hours.

`CUBLAS_STATUS_EXECUTION_FAILED` has two documented causes - the card is
out of memory, or the Windows TDR watchdog gave up on a long kernel - and
neither is a bug in the training loop. The same code trained the whole of
run 10 on the same card, at the same cap, with the same allocator.

So the fault is retried, with the allocator's cache emptied between
attempts. A fault that does not clear after the retries skips that one
batch and the run carries on, because a batch of 20,000 is worth more
than one batch of 16.
"""
import ast
import pathlib
import unittest

TRAIN_PY = pathlib.Path(__file__).resolve().parents[1] / "src" / "la" / "train.py"


class TestGpuFaultsAreRecognised(unittest.TestCase):
    def test_a_cublas_failure_is_a_gpu_fault(self):
        from la.train import is_gpu_fault
        exc = RuntimeError(
            "CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling "
            "`cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, "
            "ldb, &beta, c, ldc)`")
        self.assertTrue(is_gpu_fault(exc))

    def test_an_out_of_memory_is_a_gpu_fault(self):
        from la.train import is_gpu_fault
        self.assertTrue(is_gpu_fault(RuntimeError("CUDA out of memory.")))

    def test_a_cudnn_failure_is_a_gpu_fault(self):
        from la.train import is_gpu_fault
        self.assertTrue(is_gpu_fault(RuntimeError("cuDNN error: CUDNN_STATUS_"
                                                  "NOT_INITIALIZED")))

    def test_an_ordinary_bug_is_not_a_gpu_fault(self):
        from la.train import is_gpu_fault
        for exc in (TypeError("unsupported operand type(s)"),
                    ValueError("invalid literal for int()"),
                    KeyError("answer")):
            self.assertFalse(is_gpu_fault(exc),
                             f"{exc!r} is a bug in the code, not a fault "
                             f"in the GPU, and it must not be retried")


class TestTheTrainingLoopRetries(unittest.TestCase):
    """The retry has to be in the loop, and it has to be bounded."""

    def setUp(self):
        self.source = TRAIN_PY.read_text(encoding="utf-8")

    def test_the_loop_wraps_the_forward_and_backward_in_a_retry(self):
        self.assertIn("is_gpu_fault(exc)", self.source,
                      "the training loop does not catch a GPU fault")

    def test_the_retry_is_bounded(self):
        self.assertIn("args.gpu_retries", self.source,
                      "the retry is not bounded, so a permanent fault would "
                      "loop for ever")

    def test_a_batch_that_keeps_failing_is_skipped_not_fatal(self):
        self.assertIn("skipping this", self.source,
                      "an exhausted retry does not skip the batch")

    def test_the_retry_clears_the_allocator_cache(self):
        self.assertIn("torch.cuda.empty_cache()", self.source,
                      "the retry does not empty the cache, which is what "
                      "makes a cuBLAS fault clearable")

    def test_the_default_number_of_retries_is_more_than_one(self):
        from la.train_defaults import defaults
        self.assertGreater(defaults()["--gpu-retries"], 1)


class TestTheLoopIsStructurallySound(unittest.TestCase):
    """The retry block must not leave a name undefined.

    The loop reads `out.loss` after the retry, so a batch whose retries
    are exhausted has to skip that read rather than raise NameError.
    """

    def test_the_skip_path_exists(self):
        source = TRAIN_PY.read_text(encoding="utf-8")
        tree = ast.parse(source)
        main = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        body = ast.unparse(main)
        self.assertIn("if not ok:", body,
                      "the loop does not check whether the batch succeeded")
        self.assertIn("continue", body)


if __name__ == "__main__":
    unittest.main()
