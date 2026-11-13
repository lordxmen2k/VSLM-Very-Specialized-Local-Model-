"""Does the arithmetic printed in a scratchpad actually add up?

A scratchpad that states an expression and then gives the wrong value for it
teaches the model to fabricate. The ground truth in the corpus is computed by
sympy and is correct, so a wrong printed derivation is invisible to any test
that only compares the answer with the truth - it has to be checked against
the expression itself.

This walks every generated problem, finds every line that states an
expression and a result, evaluates the expression, and compares.

    python -m la.check_arithmetic --n 400
"""
from __future__ import annotations

import argparse
import random
import re
import sys

import sympy as sp

from .problems import GENERATORS

# a line ending in "= <number>"; the expression is what sits between the
# second-to-last "=" and that number, so prose like
# "STEP 1 expand along the first row: det(A) = (+1)*(-4) = 4" still works
ENDS_IN_NUMBER = re.compile(r"=\s*(-?[0-9]+(?:\.[0-9]+)?)\s*$")
# lines that are not arithmetic but still end in "= number"
SKIP_MARKERS = ("begin{", "\\", "det(A)", "VERIFY")


def _to_expr(text: str):
    """sympify a plain arithmetic expression, or None if it is not one."""
    text = text.strip()
    if not text or any(m in text for m in SKIP_MARKERS):
        return None
    if not re.fullmatch(r"[-+0-9*/(). ]+", text):
        return None
    try:
        expr = sp.sympify(text.replace(" ", ""))
    except Exception:
        return None
    if not expr.is_number:
        return None
    return expr


def check_problem(problem) -> list[str]:
    """Return a list of complaint strings for one problem."""
    complaints = []
    scratch = getattr(problem, "scratchpad", "") or ""
    for lineno, line in enumerate(scratch.splitlines(), start=1):
        line = line.strip()
        if not line or "=" not in line:
            continue
        m = ENDS_IN_NUMBER.search(line)
        if not m:
            continue
        rhs = m.group(1)
        head = line[:m.start()]
        cut = head.rfind("=")
        if cut == -1:
            continue
        lhs = head[cut + 1:].strip()
        expr = _to_expr(lhs)
        if expr is None:
            continue
        try:
            got = sp.nsimplify(expr)
            want = sp.nsimplify(sp.Rational(rhs) if "." in rhs else sp.Integer(rhs))
        except Exception as exc:
            complaints.append(f"{problem.family}: line {lineno}: cannot compare ({exc})")
            continue
        # Some families state a value rounded to six significant figures -
        # the singular values, and the condition number built from them - so
        # a division of two rounded numbers is not exactly the rounded
        # quotient. The tolerance is relative, and far tighter than any
        # arithmetic mistake the checker is looking for: the cofactor bug
        # this checker was written for was 8 against 4.
        diff = abs(complex(sp.N(got - want)))
        scale = max(1.0, abs(complex(sp.N(want))))
        if diff > 1e-4 * scale:
            complaints.append(
                f"{problem.family}: line {lineno}: '{lhs}' = {got}, but the "
                f"scratchpad says {rhs}")
    return complaints


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=400, help="problems per family")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    all_complaints = []
    checked = 0
    for family, gen in sorted(GENERATORS.items()):
        seen = 0
        tries = 0
        while seen < args.n and tries < args.n * 40:
            tries += 1
            p = gen(rng)
            if p is None:
                continue
            seen += 1
            checked += 1
            all_complaints.extend(check_problem(p))
        print(f"  {family:<20} {seen:>5} problems")

    print()
    print(f"checked {checked} problems across {len(GENERATORS)} families")
    if not all_complaints:
        print("every printed expression evaluates to the value it claims")
        return 0
    print(f"{len(all_complaints)} wrong expressions:")
    for c in all_complaints[:60]:
        print("  " + c)
    if len(all_complaints) > 60:
        print(f"  ... and {len(all_complaints) - 60} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
