"""Plot-spec schema and validation.

The model never draws. It emits a small declarative JSON object describing the
figure it wants, and a deterministic renderer turns that into a PNG. This keeps
the model out of the business of pixel arithmetic and makes every figure
machine-checkable: the spec carries the numbers, so a wrong spec is a wrong
answer rather than a wrong picture.

Every spec has the same three top-level keys:

    kind      one of SPEC_KINDS
    data      kind-specific numbers
    labels    axis and series labels
    annotate  list of strings to draw on the figure

`validate_spec` returns a list of human-readable problems. An empty list means
the spec is well formed. It is used in three places: when building the
training corpus, at inference time before rendering, and in the test suite.
"""
from __future__ import annotations

import math
from typing import Any

SPEC_KINDS = (
    "basis_2d",
    "basis_3d",
    "eigenvalue_spectrum",
    "row_echelon",
    "subspace_2d",
    "projection",
    "least_squares",
    "singular_spectrum",
    "matrix_heatmap",
    "transform_unit_square",
)

# Which data keys each kind requires, and how each one is shaped.
_REQUIRED: dict[str, dict[str, str]] = {
    "basis_2d": {"vectors": "matrix"},
    "basis_3d": {"vectors": "matrix3"},
    "eigenvalue_spectrum": {"values": "complex_list"},
    "row_echelon": {"matrix": "matrix", "echelon": "matrix"},
    "subspace_2d": {"vectors": "matrix", "origin": "point"},
    "projection": {"vector": "point", "projection": "point", "residual": "point",
                   "axis": "point"},
    "least_squares": {"points": "matrix", "fit": "matrix"},
    "singular_spectrum": {"values": "float_list"},
    "matrix_heatmap": {"matrix": "matrix"},
    "transform_unit_square": {"matrix": "matrix", "image": "matrix"},
}

# Keys the renderer is allowed to read. Anything else is a hallucinated key and
# is rejected before it can reach the plotting code.
_ALLOWED_DATA_KEYS = {
    "basis_2d": {"vectors"},
    "basis_3d": {"vectors"},
    "eigenvalue_spectrum": {"values"},
    "row_echelon": {"matrix", "echelon"},
    "subspace_2d": {"vectors", "origin"},
    "projection": {"vector", "projection", "residual", "axis"},
    "least_squares": {"points", "fit"},
    "singular_spectrum": {"values"},
    "matrix_heatmap": {"matrix"},
    "transform_unit_square": {"matrix", "image"},
}


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _is_point(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and len(x) == 2 and all(_is_number(c) for c in x)


def _is_matrix(x: Any) -> bool:
    if not isinstance(x, (list, tuple)) or not x:
        return False
    if not all(isinstance(row, (list, tuple)) for row in x):
        return False
    ncols = len(x[0])
    if ncols == 0 or not all(len(row) == ncols for row in x):
        return False
    return all(_is_number(c) for row in x for c in row)


def _is_matrix3(x: Any) -> bool:
    if not isinstance(x, (list, tuple)) or not x:
        return False
    if not all(isinstance(row, (list, tuple)) for row in x):
        return False
    if not all(len(row) == 3 for row in x):
        return False
    return all(_is_number(c) for row in x for c in row)


def _is_complex_list(x: Any) -> bool:
    """A list of complex numbers, each encoded as [re, im] or a plain real."""
    if not isinstance(x, (list, tuple)) or not x:
        return False
    for item in x:
        if _is_number(item):
            continue
        if isinstance(item, (list, tuple)) and len(item) == 2 and all(_is_number(c) for c in item):
            continue
        return False
    return True


def _is_float_list(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and bool(x) and all(_is_number(c) for c in x)


_SHAPES = {
    "matrix": _is_matrix,
    "matrix3": _is_matrix3,
    "point": _is_point,
    "complex_list": _is_complex_list,
    "float_list": _is_float_list,
}


def validate_spec(spec: Any) -> list[str]:
    """Return a list of problems with `spec`. Empty list means it is valid."""
    errs: list[str] = []
    if not isinstance(spec, dict):
        return [f"spec must be an object, got {type(spec).__name__}"]

    kind = spec.get("kind")
    if kind not in SPEC_KINDS:
        return [f"unknown kind {kind!r}; expected one of {', '.join(SPEC_KINDS)}"]

    for key in spec:
        if key not in ("kind", "data", "labels", "annotate"):
            errs.append(f"unexpected top-level key {key!r}")

    data = spec.get("data")
    if not isinstance(data, dict):
        errs.append("'data' must be an object")
        return errs

    allowed = _ALLOWED_DATA_KEYS[kind]
    for key in data:
        if key not in allowed:
            errs.append(f"unknown data key {key!r} for kind {kind!r}")

    for key, shape in _REQUIRED[kind].items():
        if key not in data:
            errs.append(f"missing data key {key!r} for kind {kind!r}")
            continue
        check = _SHAPES[shape]
        if not check(data[key]):
            errs.append(f"data[{key!r}] is not a valid {shape}")

    labels = spec.get("labels", {})
    if not isinstance(labels, dict):
        errs.append("'labels' must be an object")
    else:
        for k, v in labels.items():
            # a label is either one string, or a list of strings for a series
            if isinstance(v, str):
                continue
            if isinstance(v, list) and all(isinstance(s, str) for s in v):
                continue
            errs.append(f"labels[{k!r}] must be a string or a list of strings")

    annotate = spec.get("annotate", [])
    if not isinstance(annotate, list):
        errs.append("'annotate' must be a list")
    else:
        for i, s in enumerate(annotate):
            if not isinstance(s, str):
                errs.append(f"annotate[{i}] must be a string")

    return errs


def spec_features(spec: dict) -> dict:
    """Extract the numeric content of a spec for exact comparison.

    Two specs are considered to depict the same figure when their feature
    vectors match. This is how the evaluation harness compares a model's plot
    against the reference: on numbers, not on pixels.
    """
    kind = spec.get("kind")
    data = spec.get("data", {}) or {}
    feats: dict[str, Any] = {"kind": kind}
    if kind in ("basis_2d", "basis_3d"):
        feats["vectors"] = _round_all(data.get("vectors", []))
    elif kind == "eigenvalue_spectrum":
        feats["values"] = _round_all(data.get("values", []))
    elif kind == "row_echelon":
        feats["echelon"] = _round_all(data.get("echelon", []))
        feats["matrix"] = _round_all(data.get("matrix", []))
    elif kind == "subspace_2d":
        feats["vectors"] = _round_all(data.get("vectors", []))
        feats["origin"] = _round_all(data.get("origin", [0, 0]))
    elif kind == "projection":
        for k in ("vector", "projection", "residual", "axis"):
            feats[k] = _round_all(data.get(k, [0, 0]))
    elif kind == "least_squares":
        feats["points"] = _round_all(data.get("points", []))
        feats["fit"] = _round_all(data.get("fit", []))
    elif kind == "singular_spectrum":
        feats["values"] = _round_all(data.get("values", []))
    elif kind == "matrix_heatmap":
        feats["matrix"] = _round_all(data.get("matrix", []))
    elif kind == "transform_unit_square":
        feats["matrix"] = _round_all(data.get("matrix", []))
        feats["image"] = _round_all(data.get("image", []))
    return feats


def _round_all(obj: Any, ndigits: int = 6) -> Any:
    if isinstance(obj, (list, tuple)):
        return [_round_all(o, ndigits) for o in obj]
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return round(float(obj), ndigits)
    return obj


def features_match(a: dict, b: dict, tol: float = 1e-6) -> bool:
    """Compare two feature vectors with a numeric tolerance."""
    if a.get("kind") != b.get("kind"):
        return False
    for key, va in a.items():
        if key == "kind":
            continue
        vb = b.get(key)
        if not _close(va, vb, tol):
            return False
    return True


def _close(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
    return a == b
