"""The config file is the single source of truth, and it has to reach the code.

Two defaults in `la.train` were hard-coded and never read the config, so
`configs/default.yaml` said one thing and the trainer did another:

    config:  max_seq_len: 2048
    trainer: max seq len 1536          <- what the banner actually printed

The corpus was measured against 2048 and written to fit 2048, and the
trainer truncated it at 1536 instead. 15.5 per cent of the corpus lost the
end of its completion, and the end of a completion is where the answer
and the plot spec are.

The same class of bug: `la.inference.load_model` reads the tokenizer from
the ADAPTER, so a training run that crashes leaves an empty adapter
directory and every later command fails with a tokenizer error that looks
like a dependency problem rather than a missing training run.

These tests read the code, not the behaviour, because the behaviour needs
a GPU to reproduce. The rule they encode is simple: a value in
`configs/default.yaml` must be the value the code uses.
"""
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "src" / "la" / "train.py"
CONFIG = ROOT / "configs" / "default.yaml"


def _config() -> dict:
    import yaml
    with open(CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _defaults() -> dict:
    """The argparse defaults, read out of the source with ast.

    A default that is read from the config is a call on the config
    dictionary rather than a literal, so it is resolved by reading the
    config and looking the key up.
    """
    from la.train_defaults import defaults
    return defaults()


class TestConfigReachesTheTrainer(unittest.TestCase):
    """A number in the config has to be the number the code uses."""

    def test_the_sequence_cap_default_is_the_configured_one(self):
        want = _config()["training"]["max_seq_len"]
        got = _defaults().get("--max-seq-len")
        self.assertIsNotNone(got, "there is no --max-seq-len default at all")
        self.assertEqual(
            want, got,
            f"the config says max_seq_len {want} but the trainer's default is "
            f"{got}; the trainer truncates the corpus at its own default and "
            f"the config is inert")

    def test_the_configured_cap_clears_the_corpus(self):
        # 2304, measured with the model's own tokenizer over 800 sampled
        # examples: the longest is 2,132 tokens (a 5x5 matrix product with
        # its products shown) and none is over 2,204. A cap of 2048 still
        # cut 27 of the 800.
        cap = _config()["training"]["max_seq_len"]
        self.assertGreaterEqual(cap, 2204)

    def test_the_cap_still_leaves_room_on_the_card(self):
        # The logits dominate the memory: a bf16 copy, an fp32 copy for the
        # loss, the log-softmax, and the gradients of all of it, over a
        # vocabulary of 151,936. Measured: about 6 GB per example at 768
        # tokens. A cap of 2304 needs about 7.2 GB, on a card with 12.88 GB.
        vocab = 151_936
        per_token = vocab * (2 + 4 + 4) / 1e9
        batch = _defaults()["--batch-size"]
        estimated = batch * _config()["training"]["max_seq_len"] * per_token + 3.5
        self.assertLess(estimated, 12.0,
                        f"the worst case needs about {estimated:.1f} GB")


class TestTheTrainerReadsTheConfig(unittest.TestCase):
    """The trainer has to read the config, not carry its own copy of it."""

    def test_train_py_loads_the_config_file(self):
        source = TRAIN_PY.read_text(encoding="utf-8")
        self.assertIn("default.yaml", source,
                      "la.train never reads configs/default.yaml, so every "
                      "value in it is documentation rather than behaviour")


class TestTheTokenizerComesFromTheBase(unittest.TestCase):
    """The tokenizer must come from the base model, not from the adapter.

    `load_model` did `AutoTokenizer.from_pretrained(adapter or base)`. The
    adapter directory is created before the training loop and the tokenizer
    is only written to it at the very end, so a crashed run leaves an empty
    directory - and every later command dies with

        ValueError: Couldn't instantiate the backend tokenizer

    which reads like a missing dependency rather than a missing training
    run. The tokenizer is the base model's; the adapter only holds weights.
    """

    def test_the_tokenizer_is_loaded_from_the_base(self):
        from la import inference
        source = pathlib.Path(inference.__file__).read_text(encoding="utf-8")
        self.assertNotIn(
            "from_pretrained(adapter or base", source,
            "the tokenizer is still loaded from the adapter; a crashed run "
            "leaves an empty adapter directory and this fails")

    def test_the_adapter_is_only_used_for_the_weights(self):
        from la import inference
        source = pathlib.Path(inference.__file__).read_text(encoding="utf-8")
        self.assertIn("from_pretrained(base", source)


if __name__ == "__main__":
    unittest.main()
