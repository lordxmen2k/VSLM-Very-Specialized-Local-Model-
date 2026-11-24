"""Read the checkpoint with `pread` instead of `mmap` on Windows.

The failure this exists to fix:

    OSError: The paging file is too small for this operation to complete.
    (os error 1455)

raised inside `transformers`' `_load_pretrained_model`:

    file_pointer = safe_open(file, framework="pt", device=device, backend=backend)

`os error 1455` is `ERROR_COMMITMENT_LIMIT`. It is not a disk error and
not a network error: Windows could not commit enough virtual memory.

**WHY IT HAPPENS.** `safetensors` opens a shard with a copy-on-write
memory map, and Windows charges the ENTIRE view to the commit limit for
as long as the shard is open - the `MapViewOfFile` documentation says so
outright. A 3.4 GB checkpoint therefore costs 3.4 GB of commit for the
whole load, on top of the RAM, and the commit limit is RAM plus the page
file. On a machine whose page file is small or system-managed, the load
dies at the last step with an error that names the page file rather than
the real cause.

**THE FIX IS UPSTREAM.** `transformers` 5.17 switched Windows to
`safetensors`' `pread` backend, which reads each tensor into its own
buffer instead of mapping the file. Nothing is charged to commit for the
lifetime of the mapping. `safetensors` has supported the `backend`
keyword since 0.8.

**WHAT THIS DOES.** It patches `transformers.modeling_utils.safe_open` so
that, on Windows only, a CPU shard open defaults to `backend="pread"`.
The patch is a no-op everywhere else, and a no-op if `safetensors` is too
old to have the keyword or `transformers` has already been fixed.

**A LOADING FAILURE HERE IS NOT A MODEL PROBLEM.** It is the machine
running out of commit, and it says so in a way that reads like a corrupt
checkpoint. The run it broke had loaded the same model four times
earlier in the same session.
"""
from __future__ import annotations

import os
import sys


def _is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")


def install_pread_backend(verbose: bool = True) -> bool:
    """Make `transformers` read safetensors with `pread` on Windows.

    Returns whether the patch was installed. A False is not an error: it
    means the platform does not need it, or the installed versions do not
    support it, and the load will go the ordinary way.
    """
    if not _is_windows():
        return False

    try:
        import safetensors
        from safetensors import safe_open as _real_safe_open
    except Exception:
        return False

    # Does safetensors support the keyword at all? The signature does not
    # say: `safe_open` takes **kwargs, so "backend" never appears as a
    # named parameter and a signature check rejects every real install.
    # The only reliable test is to CALL it.
    try:
        _real_safe_open(os.devnull, framework="pt", device="cpu",
                        backend="pread").close()
    except TypeError:
        # safetensors is older than 0.8 and has no backend keyword
        return False
    except Exception:
        # it accepted the keyword and failed for some other reason - a
        # devnull file is not a safetensors file - which is fine
        pass

    try:
        import transformers.modeling_utils as mu
    except Exception:
        return False

    if getattr(mu, "_vslm_pread_installed", False):
        return True

    original = getattr(mu, "safe_open", None)
    if original is None:
        return False

    def safe_open_with_pread(file, framework="pt", device="cpu", **kwargs):
        """`safe_open` with the copy-on-write mapping avoided on Windows.

        The mapping is what Windows charges to the commit limit. Reading
        each tensor into its own buffer costs no commit for the lifetime
        of the file handle.
        """
        if device == "cpu" and "backend" not in kwargs:
            kwargs["backend"] = "pread"
        return original(file, framework=framework, device=device, **kwargs)

    mu.safe_open = safe_open_with_pread
    mu._vslm_pread_installed = True
    if verbose:
        print(f"  safetensors: reading with pread (safetensors "
              f"{getattr(safetensors, '__version__', '?')}), so the load "
              f"does not charge the checkpoint to the Windows commit limit")
    return True


def report_commit_limit() -> None:
    """Print the commit limit, if this platform can report it.

    Windows exposes the commit limit through the performance counters.
    It is the number that decides whether a 3.4 GB checkpoint can be
    mapped at all, so it is worth seeing before a long run rather than
    after a failure.
    """
    if not _is_windows():
        return
    try:
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total = stat.ullTotalPageFile / 1e9
        avail = stat.ullAvailPageFile / 1e9
        ram = stat.ullTotalPhys / 1e9
        print(f"  commit limit    {total:.1f} GB "
              f"(RAM {ram:.1f} GB + page file), {avail:.1f} GB free")
        if avail < 6.0:
            print("  WARNING: under about 6 GB of commit is free. A 3.4 GB")
            print("  checkpoint cannot be mapped here, and the load will")
            print("  die with 'the paging file is too small' (os error")
            print("  1455). Close what is holding memory, or enlarge the")
            print("  page file.")
    except Exception as exc:
        print(f"  could not read the commit limit: {exc}")
