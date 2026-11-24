"""A checkpoint must not be charged to the Windows commit limit.

The failure:

    OSError: The paging file is too small for this operation to complete.
    (os error 1455)

raised inside `transformers`' `_load_pretrained_model`:

    file_pointer = safe_open(file, framework="pt", device=device, backend=backend)

`os error 1455` is `ERROR_COMMITMENT_LIMIT`. It is not a corrupt
checkpoint, and it is not a disk error: Windows could not commit enough
virtual memory.

**WHY.** `safetensors` opens a shard with a copy-on-write memory map, and
Windows charges the ENTIRE view to the commit limit for as long as the
shard is open. The `MapViewOfFile` documentation says so outright:

    "When copy-on-write access is specified, the system and process
    commit charge taken is for the entire view"

So a 3.4 GB checkpoint costs 3.4 GB of commit for the whole load, on top
of the RAM, and the commit limit is RAM plus the page file.

**THE FIX IS UPSTREAM.** `transformers` 5.17 switched Windows to
`safetensors`' `pread` backend, which reads each tensor into its own
buffer and maps nothing. `safetensors` has had the `backend` keyword
since 0.8.

These tests check the patch, not the machine: the sandbox has no GPU
stack, so what can be checked here is that the patch is installed where
it should be, is a no-op where it should be, and does not break a load.
"""
import ast
import sys
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestThePreadModule(unittest.TestCase):
    def test_it_exists_and_exports_the_two_functions(self):
        from la import pread
        self.assertTrue(callable(pread.install_pread_backend))
        self.assertTrue(callable(pread.report_commit_limit))

    def test_it_documents_the_failure_it_exists_for(self):
        from la import pread
        source = pathlib.Path(pread.__file__).read_text(encoding="utf-8")
        self.assertIn("1455", source,
                      "the module must name the error it fixes, so the next "
                      "person who sees it knows what to look for")

    def test_the_commit_limit_is_reported(self):
        from la import pread
        source = pathlib.Path(pread.__file__).read_text(encoding="utf-8")
        self.assertIn("GlobalMemoryStatusEx", source,
                      "the commit limit is not reported, so the failure is "
                      "still a surprise")

    def test_the_warning_names_the_fix(self):
        from la import pread
        source = pathlib.Path(pread.__file__).read_text(encoding="utf-8")
        self.assertIn("page file", source)


class TestThePatchIsWiredIn(unittest.TestCase):
    """Both places that load a checkpoint have to use it."""

    def test_the_inference_path_installs_it(self):
        source = (ROOT / "src" / "la" / "inference.py").read_text(
            encoding="utf-8")
        self.assertIn("install_pread_backend", source,
                      "the scoring path loads a 3.4 GB checkpoint and does "
                      "not install the pread backend")

    def test_the_training_path_installs_it(self):
        source = (ROOT / "src" / "la" / "train.py").read_text(
            encoding="utf-8")
        self.assertIn("install_pread_backend", source,
                      "the training path loads a 3.4 GB checkpoint and does "
                      "not install the pread backend")


class TestThePatchIsSafe(unittest.TestCase):
    """A patch that breaks the load would be worse than no patch."""

    def test_it_is_a_no_op_off_windows(self):
        from la import pread
        # the function returns False on a platform that does not need it,
        # and it must not raise on any platform
        try:
            result = pread.install_pread_backend(verbose=False)
        except Exception as exc:
            self.fail(f"install_pread_backend raised: {exc}")
        self.assertIsInstance(result, bool)
        if not pread._is_windows():
            self.assertFalse(result,
                             "the patch must do nothing off Windows")

    def test_it_never_raises_when_the_stack_is_missing(self):
        # the sandbox has no torch, transformers or safetensors, so the
        # patch has to be a clean no-op there
        from la import pread
        self.assertIsInstance(pread.install_pread_backend(verbose=False), bool)

    def test_the_wrapped_open_keeps_the_backend_when_one_is_given(self):
        """A caller that asks for a backend must get it."""
        from la import pread
        source = pathlib.Path(pread.__file__).read_text(encoding="utf-8")
        self.assertIn('"backend" not in kwargs', source,
                      "the wrapper overrides an explicitly requested backend")

    def test_the_wrapper_only_uses_pread_for_cpu(self):
        from la import pread
        source = pathlib.Path(pread.__file__).read_text(encoding="utf-8")
        self.assertIn('device == "cpu"', source,
                      "a device-resident open must not be switched to pread")


class TestThePatchActuallyInstalls(unittest.TestCase):
    """A patch that silently does not install is worse than no patch.

    The first version of this check asked whether `safe_open`'s signature
    had a `backend` parameter. It does not - the function takes
    **kwargs - so the check rejected every real installation and the
    patch never went in, quietly, on every machine.

    So the test builds a fake safetensors and a fake transformers and
    checks that the patch is really installed and really changes what is
    passed to the open.
    """

    def _fake_stack(self, backend_supported=True):
        import types
        calls = []

        def real_safe_open(file, framework="pt", device="cpu", **kw):
            if backend_supported and "backend" in kw:
                pass
            elif "backend" in kw:
                raise TypeError("safe_open() got an unexpected keyword "
                                "argument 'backend'")
            calls.append({"file": file, "device": device, **kw})

            class Handle:
                def close(self_inner):
                    pass
            return Handle()

        st = types.ModuleType("safetensors")
        st.__version__ = "0.8.0"
        st.safe_open = real_safe_open
        mu = types.ModuleType("transformers.modeling_utils")
        mu.safe_open = real_safe_open
        tf = types.ModuleType("transformers")
        tf.modeling_utils = mu
        sys.modules["safetensors"] = st
        sys.modules["transformers"] = tf
        sys.modules["transformers.modeling_utils"] = mu
        return mu, calls

    def setUp(self):
        self._saved = {k: sys.modules.get(k)
                       for k in ("safetensors", "transformers",
                                 "transformers.modeling_utils")}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_the_patch_is_installed_and_changes_the_open(self):
        from la import pread
        mu, calls = self._fake_stack()
        was_windows = pread._is_windows
        pread._is_windows = lambda: True
        try:
            installed = pread.install_pread_backend(verbose=False)
        finally:
            pread._is_windows = was_windows
        self.assertTrue(installed,
                        "the patch did not install; the signature check "
                        "must not be how safetensors' support is detected")

        calls.clear()
        mu.safe_open("m.safetensors", framework="pt", device="cpu")
        self.assertEqual("pread", calls[-1].get("backend"),
                         "a CPU open is not using pread, so the checkpoint is "
                         "still charged to the Windows commit limit")

    def test_a_device_open_is_left_alone(self):
        from la import pread
        mu, calls = self._fake_stack()
        was_windows = pread._is_windows
        pread._is_windows = lambda: True
        try:
            pread.install_pread_backend(verbose=False)
        finally:
            pread._is_windows = was_windows
        calls.clear()
        mu.safe_open("m.safetensors", framework="pt", device="cuda:0")
        self.assertNotIn("backend", calls[-1])

    def test_an_explicit_backend_is_respected(self):
        from la import pread
        mu, calls = self._fake_stack()
        was_windows = pread._is_windows
        pread._is_windows = lambda: True
        try:
            pread.install_pread_backend(verbose=False)
        finally:
            pread._is_windows = was_windows
        calls.clear()
        mu.safe_open("m.safetensors", framework="pt", device="cpu",
                     backend="mmap")
        self.assertEqual("mmap", calls[-1].get("backend"))

    def test_installing_twice_does_not_double_wrap(self):
        from la import pread
        mu, calls = self._fake_stack()
        was_windows = pread._is_windows
        pread._is_windows = lambda: True
        try:
            pread.install_pread_backend(verbose=False)
            pread.install_pread_backend(verbose=False)
        finally:
            pread._is_windows = was_windows
        calls.clear()
        mu.safe_open("m.safetensors", device="cpu")
        self.assertEqual("pread", calls[-1].get("backend"))


class TestThePatchDoesNotBreakTheModule(unittest.TestCase):
    def test_every_module_compiles(self):
        import la.pread
        path = pathlib.Path(la.pread.__file__)
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


if __name__ == "__main__":
    unittest.main()
