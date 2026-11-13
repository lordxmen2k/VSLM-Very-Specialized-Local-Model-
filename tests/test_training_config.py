"""The training configuration has to stay what the README says it is.

The defaults were wrong once already: batch 1 with 16-way accumulation and
2048-token sequences measured 9.0 s per optimiser step, which is six hours
for two epochs, while the README claimed thirty minutes for the whole run.
That number had never been measured.
"""
import ast
import pathlib
import unittest

TRAIN_PY = pathlib.Path(__file__).resolve().parents[1] / "src" / "la" / "train.py"


def _defaults() -> dict:
    """The argparse defaults, read out of the source with ast."""
    tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", "") != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        flag = node.args[0].value
        for kw in node.keywords:
            if kw.arg == "default":
                found[flag] = ast.literal_eval(kw.value)
    return found


class TestTrainingDefaults(unittest.TestCase):
    def setUp(self):
        self.d = _defaults()

    def test_effective_batch_is_sixteen(self):
        bs = self.d["--batch-size"]
        ga = self.d["--grad-accum"]
        self.assertEqual(16, bs * ga,
                         f"effective batch is {bs} x {ga} = {bs*ga}, not 16")

    def test_micro_batch_is_the_measured_optimum(self):
        # make speed-test on an RTX 4070 SUPER measured five shapes at the
        # same effective batch of 16:
        #
        #   batch  accum   s/step     tok/s   reserved
        #       1     16      6.87       814       4.33   under the card
        #       2      8      3.58     1,633       7.45   under the card
        #       4      4     24.40       231      13.68   over it
        #       8      2     36.36       169      26.12   over it
        #      16      1     29.04       191      27.97   over it
        #
        # A configuration whose reserved memory passes the card's physical
        # VRAM spills into host memory over PCIe and is several times
        # slower. Batch 2 was the fastest row that stayed under it.
        self.assertEqual(2, self.d["--batch-size"])
        self.assertEqual(8, self.d["--grad-accum"])

    def test_sequence_length_leaves_the_tail_intact(self):
        # the corpus runs to a median of 344 tokens with a tail past 512, so
        # a 512 cap was cutting a few per cent of the examples and losing
        # the end of their completions. The cap has to clear the tail, and
        # make speed-test reports how many examples it is still cutting.
        self.assertGreaterEqual(self.d["--max-seq-len"], 512)
        self.assertLessEqual(self.d["--max-seq-len"], 1024)

    def test_the_worst_case_logits_leave_room_on_the_card(self):
        # the logits dominate the memory: a bf16 copy, an fp32 copy for the
        # loss, and the gradient of that copy, over a vocabulary of 151,936.
        vocab = 151_936
        per_token = vocab * (2 + 4 + 4) / 1e9
        logits_gb = self.d["--batch-size"] * self.d["--max-seq-len"] * per_token
        # measured: batch 2 at a 512 cap allocated 5.04 GB, of which the
        # logits are about 1.6 GB
        estimated = logits_gb + 3.5
        self.assertLess(estimated, 12.0,
                        f"the worst case needs about {estimated:.1f} GB")

    def test_default_is_one_epoch(self):
        self.assertEqual(1.0, self.d["--epochs"])


class TestDatasetIsTokenisedOnce(unittest.TestCase):
    def test_tokenizer_is_not_called_during_training(self):
        try:
            from la.train import ProblemDataset
        except Exception as exc:          # torch or transformers missing
            self.skipTest(f"cannot import la.train: {exc}")

        calls = []

        def fake_tokenizer(text, add_special_tokens=False):
            calls.append(text)
            return {"input_ids": list(range(len(text.split())))}

        rows = [{"question": "q one", "scratchpad": "s one", "answer": "1",
                 "spec": {}},
                {"question": "q two", "scratchpad": "s two", "answer": "2",
                 "spec": {}}]
        ds = ProblemDataset(rows, fake_tokenizer, max_len=64)
        n_at_build = len(calls)
        self.assertEqual(2, n_at_build, "the corpus should be tokenised once, "
                                        "up front")
        for i in range(len(ds)):
            ds[i]
        self.assertEqual(n_at_build, len(calls),
                         "the tokenizer must not be called again on access")


if __name__ == "__main__":
    unittest.main()


class TestSpeedTestParent(unittest.TestCase):
    """The parent process must not hold GPU memory while the children measure.

    It loads the tokenizer and spawns a process per configuration. If it also
    loads the model, the card is shared with the parent's copy while the
    child measures, and every row is measured against a card that is already
    partly full. That is not a subtle effect: it turned a 4.3 GB row into a
    9.8 GB one and made the fast configuration look like the slow one.
    """

    def test_the_speed_test_branch_comes_before_the_model_is_built(self):
        import ast
        tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))
        main = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        speed_at = build_at = None
        for i, node in enumerate(main.body):
            if isinstance(node, ast.If) and "args.speed_test" in ast.unparse(node.test):
                speed_at = i
            if isinstance(node, ast.Assign) and "build_model" in ast.unparse(node):
                build_at = i
        self.assertIsNotNone(speed_at, "main() has no --speed-test branch")
        self.assertIsNotNone(build_at, "main() never builds the model")
        self.assertLess(speed_at, build_at,
                        "the speed test must return before the model is built, "
                        "or the parent holds the GPU while its children measure")


class TestSpeedRowParsing(unittest.TestCase):
    """The speed test measures each configuration in its own process and
    reads the row back over stdout, so the row format has to round-trip."""

    def test_a_row_parses(self):
        from la.speed import format_row, parse_speed_row
        line = format_row(2, 8, 4.25, 1454.0, 0.5, 6.41, 10.86, 12.88)
        row = parse_speed_row(line)
        self.assertIsNotNone(row)
        self.assertFalse(row["oom"])
        self.assertEqual(2, row["bs"])
        self.assertEqual(8, row["ga"])
        self.assertEqual(16, row["eff"])
        self.assertAlmostEqual(4.25, row["s_step"])
        self.assertAlmostEqual(1454.0, row["tok_s"])
        self.assertAlmostEqual(6.41, row["alloc"])
        self.assertAlmostEqual(10.86, row["resv"])
        self.assertAlmostEqual(12.88, row["free_start"])

    def test_an_oom_row_parses(self):
        from la.speed import parse_speed_row
        self.assertEqual({"oom": True}, parse_speed_row("SPEEDROW|oom"))

    def test_junk_does_not_parse(self):
        from la.speed import parse_speed_row
        for bad in (None, "", "not a row", "SPEEDROW|2|8", "SPEEDROW|x|y|z"):
            self.assertIsNone(parse_speed_row(bad))
