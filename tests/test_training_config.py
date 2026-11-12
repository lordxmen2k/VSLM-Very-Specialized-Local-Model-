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

    def test_micro_batches_are_not_one(self):
        # batch 1 with 16-way accumulation is what made the run six hours
        self.assertGreater(self.d["--batch-size"], 1,
                           "a batch of one leaves the GPU under-occupied")

    def test_sequence_length_is_bounded(self):
        # the corpus runs to about 450 tokens; 2048 wastes memory on the
        # logits and buys nothing
        self.assertLessEqual(self.d["--max-seq-len"], 512)

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
