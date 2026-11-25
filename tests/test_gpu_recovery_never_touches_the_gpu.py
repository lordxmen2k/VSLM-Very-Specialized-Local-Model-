"""The GPU recovery path must not itself need the GPU.

Run 14 died at step 520 of 1,250:

    GPU fault (AcceleratorError); clearing the cache and retrying (attempt 1)
    ...
    torch.AcceleratorError: CUDA error: out of memory

    During handling of the above exception, another exception occurred:

    File "src\\la\\train.py", line 755, in main
        win_tok -= int(batch["attention_mask"].sum().item())
    torch.AcceleratorError: CUDA error: out of memory

The retry machinery worked: the fault was recognised as a GPU fault, not a
bug in the code, and the cache was cleared. Then the recovery path called
`.item()` on a tensor that had already been moved to the GPU. `.item()`
copies a value off the device, and the device had just run out of memory -
so the cure for the fault raised the fault again, this time outside the
`try` block that was supposed to catch it. A three-and-a-half-hour run
died at step 520 because of a bookkeeping line.

The rule: while the card is in an unknown state, the recovery path must
not ask the card for anything. The token count is a throughput
counter, it is only ever used to print a tok/s figure, and it can be
computed on the CPU - where it costs nothing - before the batch is
moved to the device.

The other half of the defect is the accounting itself. The batch's
token count is added to the window counter before the forward pass and
subtracted on the retry. If the retry happens, the batch is run again
and the count is added again, so the counter is right; but if the
count is taken from the GPU tensor, the subtraction is the thing that
kills the run.
"""
import ast
import pathlib
import unittest

TRAIN_PY = pathlib.Path(__file__).resolve().parents[1] / "src" / "la" / "train.py"


def _torch_or_skip():
    """torch, or skip the test.

    The count helper is duck-typed - it only needs `.sum()` and
    `.item()` - so a plain list exercises it too, and that test runs
    anywhere. The tensor tests need torch, which is not installed on
    every machine that runs this suite.
    """
    try:
        import torch
    except ImportError:
        raise unittest.SkipTest("torch is not installed here")
    return torch


def _main_ast():
    tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))
    return next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")


def _module_ast():
    return ast.parse(TRAIN_PY.read_text(encoding="utf-8"))


def _code_without_docstrings(node) -> str:
    """The source of a function, with its docstring removed.

    The docstring of `run_batch_with_retries` quotes the line that
    killed run 14, so a test that reads the source and looks for
    `.item()` finds it in the prose. The prose is not the code.
    """
    import copy
    node = copy.deepcopy(node)
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef, ast.Module)):
            body = getattr(child, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                rest = body[1:]
                if not rest:
                    # a body of only a docstring: leave one statement
                    # behind so the tree is still well formed
                    placeholder = ast.Pass()
                    ast.copy_location(placeholder, body[0])
                    rest = [placeholder]
                child.body = rest
    return ast.unparse(node)


class TestTheRecoveryPathNeverTouchesTheDevice(unittest.TestCase):
    """No `.item()` on a device tensor may appear inside the retry block."""

    def setUp(self):
        self.main = _main_ast()
        self.body = _code_without_docstrings(self.main)

    def _retry_block(self):
        """The source of the retry loop that catches a GPU fault.

        The retry lives in `run_batch_with_retries`, a function of its
        own so that it can be tested without a GPU. The epoch loop in
        `main` also mentions `is_gpu_fault(exc)` in its subtree, so the
        outermost `while` is the wrong one - it also contains the
        success path, where `out.loss.item()` is correct and necessary.
        """
        for node in ast.walk(_module_ast()):
            if isinstance(node, ast.FunctionDef) and \
                    node.name == "run_batch_with_retries":
                return _code_without_docstrings(node)
        self.fail("there is no run_batch_with_retries function")

    def test_the_retry_block_does_not_call_item_on_the_attention_mask(self):
        block = self._retry_block()
        self.assertNotIn('batch["attention_mask"].sum().item()', block,
                         "the recovery path reads a value off the GPU, and the "
                         "GPU is what just ran out of memory")

    def test_the_retry_block_has_no_item_call_at_all(self):
        """Any `.item()` in the recovery path is a device round-trip."""
        block = self._retry_block()
        self.assertNotIn(".item()", block,
                         "the recovery path calls .item(), which copies a "
                         "value off the device; it must not touch the device")

    def test_the_token_count_is_taken_before_the_batch_moves_to_the_gpu(self):
        """The count is computed on the CPU, from the CPU batch."""
        lines = self.body.splitlines()
        moved = [i for i, l in enumerate(lines) if "move_batch_to_device" in l]
        counted = [i for i, l in enumerate(lines)
                   if "count_batch_tokens(batch)" in l]
        self.assertTrue(moved, "the batch is not moved to the device at all")
        self.assertTrue(counted, "the batch's token count is never taken")
        self.assertLess(min(counted), max(moved),
                        "the token count is taken after the batch has already "
                        "been moved to the GPU, so it is a device read")

    def test_the_count_happens_before_the_transfer(self):
        """The count is taken from the CPU batch, not from the moved one."""
        lines = self.body.splitlines()
        moved = [i for i, l in enumerate(lines) if "move_batch_to_device" in l]
        self.assertTrue(moved)
        first_move = moved[0]
        for l in lines[:first_move]:
            if "count_batch_tokens(batch)" in l:
                return
        self.fail("the token count is not taken from the CPU batch")

    def test_the_retry_subtracts_the_cpu_count_not_a_device_read(self):
        """The retry path is handed the count, not the device.

        The token counter is corrected by a callback the caller
        supplies, because the recovery path may not read the batch.
        The caller computes the count on the CPU before the batch moves.
        """
        block = self._retry_block()
        self.assertNotIn("attention_mask", block,
                         "the retry reads the attention mask, which lives "
                         "on the GPU by then")
        self.assertIn("on_retry", block,
                      "the retry does not report that it recovered, so the "
                      "caller cannot correct its token counter")
        # and the caller corrects it from the CPU count
        self.assertIn("batch_tokens", self.body,
                      "the caller does not correct the token counter")
        self.assertIn("on_retry=_correct_the_counter", self.body,
                      "the retry callback is not wired up")


class TestTheRecoveryPathIsTestableWithoutAGpu(unittest.TestCase):
    """The recovery must be exercisable without a GPU at all.

    A defect that can only be seen on a machine with a 12 GB card and
    three and a half hours of training is a defect that ships. The
    counting has to be a function that can be called on a CPU tensor -
    which means it cannot live in `la.train`, because that module
    imports torch at module level.
    """

    def test_the_helper_lives_where_torch_is_not_needed(self):
        from la import train_defaults
        self.assertTrue(hasattr(train_defaults, "count_batch_tokens"),
                        "the token counter is not in a module that can be "
                        "imported without torch")

    def test_there_is_a_helper_that_counts_the_tokens_of_a_batch(self):
        torch = _torch_or_skip()
        from la.train_defaults import count_batch_tokens
        batch = {
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        }
        self.assertEqual(count_batch_tokens(batch), 5)

    def test_the_helper_works_on_a_cpu_tensor_without_a_device(self):
        torch = _torch_or_skip()
        from la.train_defaults import count_batch_tokens
        batch = {"attention_mask": torch.tensor([[1, 0, 1]])}
        self.assertIsInstance(count_batch_tokens(batch), int)
        self.assertEqual(count_batch_tokens(batch), 2)

    def test_the_helper_copes_with_a_missing_attention_mask(self):
        from la.train_defaults import count_batch_tokens
        self.assertEqual(count_batch_tokens({}), 0)

    def test_the_helper_works_with_a_plain_list(self):
        """It does not need to be a tensor at all."""
        from la.train_defaults import count_batch_tokens
        batch = {"attention_mask": [[1, 1, 0], [1, 0, 0]]}
        self.assertEqual(count_batch_tokens(batch), 3)


class TestTheEvaluationPathRefusesAMissingAdapter(unittest.TestCase):
    """A missing adapter must be an error, not a hub download.

    After run 14's training died, `make eval` was still run. It tried
    to load `runs/vslm-run14` as a Hugging Face repository:

        httpx.HTTPStatusError: Client error '404 Not Found' for url
        'https://huggingface.co/runs/vslm-run14/resolve/main/adapter_config.json'

    `la.bench` already refuses an empty adapter directory with a clear
    message. `la.evaluate` does not: it checks only that the path is a
    directory, and a directory that exists but holds no adapter is
    passed to PEFT, which tries to fetch it from the Hub and produces a
    404 with no hint that the real problem is a training run that did
    not finish.
    """

    EVALUATE_PY = (pathlib.Path(__file__).resolve().parents[1]
                   / "src" / "la" / "evaluate.py")

    def setUp(self):
        self.source = self.EVALUATE_PY.read_text(encoding="utf-8")

    def test_evaluate_calls_the_adapter_guard(self):
        self.assertIn("_check_adapter_base", self.source,
                      "evaluate does not run the adapter guard, so a "
                      "directory with no adapter in it is loaded as if it "
                      "were one")

    def test_the_guard_runs_before_the_model_is_loaded(self):
        """A guard after the load would be too late."""
        tree = ast.parse(self.source)
        main = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        body = ast.unparse(main)
        self.assertLess(body.index("_check_adapter_base"),
                        body.index("load_model"),
                        "the adapter is checked after the model is loaded")

    def test_the_guard_is_the_same_one_the_benchmark_uses(self):
        """One rule, not two that can drift apart."""
        tree = ast.parse(self.source)
        main = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        body = ast.unparse(main)
        self.assertIn("from .bench import _check_adapter_base", body,
                      "evaluate does not reuse the benchmark's guard")

    def test_the_guard_holds_the_empty_directory_rule(self):
        """The guard evaluate calls is the one that refuses an empty dir.

        The rule itself is covered in test_empty_adapter_is_refused.py;
        what is new here is that evaluate calls it at all.
        """
        bench = (pathlib.Path(__file__).resolve().parents[1]
                 / "src" / "la" / "bench.py").read_text(encoding="utf-8")
        self.assertIn("REFUSING TO RUN", bench,
                      "the guard no longer refuses an empty adapter")


if __name__ == "__main__":
    unittest.main()
