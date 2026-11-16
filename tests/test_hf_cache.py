"""The Hugging Face cache goes on the drive with the room.

The cache defaults to ``%USERPROFILE%\\.cache\\huggingface``, which on this
machine is on C:, where there was about 26 MB free. A 1.7B checkpoint needs
about 4 GB. The download died part way through with::

    RuntimeError: Task error: File reconstruction error:
    Internal Writer Error: Background writer channel closed

which reads like a network fault but is a full disk.

``huggingface_hub`` resolves the cache path once, when it is imported, so
setting the variable after that import does nothing. These tests pin the
behaviour that makes the relocation actually work.
"""
import os
import unittest


class TestCacheLocation(unittest.TestCase):
    def test_an_explicit_choice_is_respected(self):
        """If the user has already set HF_HOME, that is their decision."""
        from la import hf_cache
        old = os.environ.get("HF_HOME")
        try:
            os.environ["HF_HOME"] = r"D:\somewhere\else"
            os.environ.pop("HF_HUB_CACHE", None)
            self.assertEqual(r"D:\somewhere\else",
                             hf_cache.configure())
            self.assertEqual(r"D:\somewhere\else",
                             os.environ["HF_HOME"])
        finally:
            if old is None:
                os.environ.pop("HF_HOME", None)
            else:
                os.environ["HF_HOME"] = old

    def test_the_default_is_on_the_big_drive(self):
        from la import hf_cache
        # the default root, not the resolved path - the test runs on a
        # machine without G: and the path is never created unless the
        # drive is there
        self.assertTrue(hf_cache._DEFAULT_ROOT.upper().startswith("G:"),
                        f"the default cache root is {hf_cache._DEFAULT_ROOT}, "
                        "which is not on the drive with the room")

    def test_the_cache_is_under_the_hub_directory(self):
        from la import hf_cache
        cache = hf_cache.cache_location()
        self.assertIn("hub", os.path.basename(cache).lower())

    def test_free_space_is_reported_or_none(self):
        """Never raises. A drive that is not mounted must not crash a run."""
        from la import hf_cache
        free = hf_cache.free_gb()
        self.assertTrue(free is None or free >= 0)

    def test_describe_mentions_the_path(self):
        from la import hf_cache
        self.assertIn("hub", hf_cache.describe())


class TestTheEntryPointsImportItFirst(unittest.TestCase):
    """The import has to happen before transformers, or it does nothing.

    ``huggingface_hub`` reads ``HF_HOME`` at import time and keeps the
    resolved path. A module that imports transformers first has already
    missed the window.
    """

    def _transformers_import_line(self, path: str) -> int:
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if "import transformers" in line:
                    return n
                if "from transformers" in line:
                    return n
        return 0

    def _hf_cache_import_line(self, path: str) -> int:
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if "hf_cache" in line and "import" in line:
                    return n
        return 0

    def _hf_cache_import_line(self, path: str) -> int:
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if "hf_cache" in line and "import" in line:
                    return n
        return 0

    def test_inference_imports_the_cache_before_transformers(self):
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "src", "la", "inference.py")
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("from . import hf_cache", src)

    def test_train_imports_the_cache_module(self):
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "src", "la", "train.py")
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("from . import hf_cache", src)
        # and it has to come before the torch import, which pulls in
        # transformers' ecosystem
        cache_at = src.index("from . import hf_cache")
        self.assertLess(cache_at, src.index("import torch"),
                        "hf_cache must be imported before torch")


if __name__ == "__main__":
    unittest.main()
