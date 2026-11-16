"""Put the Hugging Face cache on a drive with room, before anything imports it.

The cache defaults to ``%USERPROFILE%\\.cache\\huggingface``. On this machine
that is on C:, which has room for about 26 MB, and a 1.7B checkpoint needs
about 4 GB. A download that runs out of room part way through dies with::

    RuntimeError: Task error: File reconstruction error:
    Internal Writer Error: Background writer channel closed

which reads like a network fault but is a full disk.

``huggingface_hub`` reads ``HF_HOME`` once, when it is imported, and caches
the resolved path. Setting the variable after that import has no effect, so
this module has to run first. Import it at the top of anything that loads a
model, before ``transformers`` or ``huggingface_hub``.

    from la.hf_cache import cache_location
    print(cache_location())

To make it permanent for every shell, run ``scripts\\setup_hf_home.bat``.
"""
from __future__ import annotations

import os
import sys

# The drive with the room. Override with the VSLM_HF_HOME environment
# variable if the models live somewhere else.
_DEFAULT_ROOT = r"G:\models"

# Where the cache is put when the environment does not already say.
_HOME_ENV = "HF_HOME"
_CACHE_ENV = "HF_HUB_CACHE"


def _default_home() -> str:
    root = os.environ.get("VSLM_HF_HOME", _DEFAULT_ROOT)
    return os.path.join(root, "huggingface")


def configure() -> str:
    """Point the cache at a drive with room. Returns the cache root.

    Does nothing if ``HF_HOME`` is already set - an explicit choice is
    respected. Otherwise the default location is used, and it is created if
    it is not there.
    """
    home = os.environ.get(_HOME_ENV)
    if not home:
        home = _default_home()
        os.environ[_HOME_ENV] = home
    cache = os.environ.get(_CACHE_ENV)
    if not cache:
        cache = os.path.join(home, "hub")
        os.environ[_CACHE_ENV] = cache
    try:
        os.makedirs(cache, exist_ok=True)
    except OSError:
        # A drive that is not mounted, or a path that cannot be created. The
        # default cache is still better than nothing, so leave the
        # environment as it is and let the caller find out.
        pass
    return home


def cache_location() -> str:
    """The directory the Hugging Face cache is using."""
    home = os.environ.get(_HOME_ENV) or _default_home()
    return os.environ.get(_CACHE_ENV) or os.path.join(home, "hub")


def free_gb(path: str | None = None) -> float | None:
    """Free space on the drive the cache is on, in gigabytes."""
    import shutil
    target = path or cache_location()
    probe = target
    try:
        while probe and not os.path.isdir(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        return shutil.disk_usage(probe).free / 1e9
    except Exception:
        return None


def describe() -> str:
    """One line about where the cache is and how much room is left."""
    loc = cache_location()
    free = free_gb(loc)
    if free is None:
        return f"{loc} (free space unknown)"
    return f"{loc} ({free:.1f} GB free)"


# Configure on import, so that a module which imports this before
# huggingface_hub gets the right answer. The import is guarded because
# la.hf_cache must stay importable without the rest of the stack.
configure()


if __name__ == "__main__":
    print(f"HF_HOME      = {os.environ.get(_HOME_ENV)}")
    print(f"HF_HUB_CACHE = {os.environ.get(_CACHE_ENV)}")
    print(f"cache        = {describe()}")
    print(f"python       = {sys.executable}")
