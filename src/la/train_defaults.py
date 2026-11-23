"""Read the trainer's argparse defaults, including the ones from the config.

Most of the trainer's defaults are literals, but some of them are read
from `configs/default.yaml` - the sequence cap is one, and it was
hard-coded at 1536 in the trainer while the config said 2048. A test that
reads the defaults has to resolve both kinds, and the two kinds are
resolved differently: a literal is evaluated, and a config-backed default
is looked up in the config file.

    from la.train_defaults import defaults
    d = defaults()
    d["--max-seq-len"]
"""
from __future__ import annotations

import ast
import os

# this file lives in src/la, so the repository root is two levels up
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_PY = os.path.join(_ROOT, "src", "la", "train.py")
CONFIG_YAML = os.path.join(_ROOT, "configs", "default.yaml")

# the name the default is read from, and the section of the config it
# comes from. `cfg` is the whole file, so it has no section.
_SECTIONS = {"training": "training", "model_cfg": "model", "cfg": None}


def _config() -> dict:
    import yaml
    with open(CONFIG_YAML, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _config_default(node) -> object:
    """Resolve `section.get("key", fallback)` against the config file."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "get":
        return None
    if not isinstance(func.value, ast.Name):
        return None
    section = func.value.id
    if section not in _SECTIONS:
        return None
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return None
    key = node.args[0].value
    fallback = None
    if len(node.args) > 1:
        try:
            fallback = ast.literal_eval(node.args[1])
        except (ValueError, TypeError):
            fallback = None
    if _SECTIONS[section] is None:
        return fallback
    return _config().get(_SECTIONS[section], {}).get(key, fallback)


def defaults(path: str | None = None) -> dict:
    """The argparse defaults of la.train, keyed by flag."""
    with open(path or TRAIN_PY, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    found: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", "") != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        flag = node.args[0].value
        for kw in node.keywords:
            if kw.arg != "default":
                continue
            try:
                found[flag] = ast.literal_eval(kw.value)
            except (ValueError, TypeError):
                found[flag] = _config_default(kw.value)
    return found
