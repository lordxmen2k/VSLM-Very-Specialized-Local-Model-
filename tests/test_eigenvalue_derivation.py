"""The eigenvalue derivation must be valid LaTeX and must show its work.

TWO DEFECTS THIS FILE GUARDS AGAINST.

**1. The characteristic polynomial was written in sympy's PLAIN-TEXT
form.** Every one of the corpus's eigenvalue derivations said

    the characteristic polynomial is p(λ) = lambda**3 + 7*lambda**2 - 38*lambda - 240

`lambda**3` is Python syntax, not LaTeX. In LaTeX it renders as
"lambda**3" in italic text - the model is trained to emit a broken
expression and then has to parse one at benchmark time. The
characteristic polynomial is the FIRST line of the derivation, so every
eigenvalue example was malformed at its first line.

**2. The roots were ASSERTED, not computed.** The line read

    the roots of p are -5, 6, -8, so the eigenvalues of A are { -5, 6, -8 }

which states the answer without showing the factoring that produced it.
The model was taught to emit a set of numbers it could not compute -
the same defect as the determinant's asserted minors, and the same
lesson as run 9's dropped dot-product products.

The eigenvalue family scored 1 of 30 at 3x3 against the base model's
2 of 30, and the corpus taught a malformed derivation for all of them.
"""
from __future__ import annotations

import random
import re
import unittest

from la.problems import generate


def _eigenvalues(family="eigenvalue", n=60, seed=4242):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = generate(family, rng)
        if p is not None:
            out.append(p)
    return out


class TestTheEigenvalueDerivationIsValidLatex(unittest.TestCase):

    def test_the_characteristic_polynomial_is_latex_not_python(self):
        """`lambda**3` is Python. The corpus must write `\\lambda^{3}`."""
        checked = 0
        for p in _eigenvalues():
            m = re.search(r"p\(λ\) = (.+)$", p.scratchpad, re.M)
            if not m:
                continue
            checked += 1
            poly = m.group(1)
            self.assertNotIn(
                "**", poly,
                f"the characteristic polynomial is in Python's syntax: "
                f"{poly!r}")
            self.assertNotIn(
                "lambda**", poly,
                f"the characteristic polynomial does not use a backslash: "
                f"{poly!r}")
            self.assertIn(
                "\\lambda", poly,
                f"the characteristic polynomial does not use a backslash: "
                f"{poly!r}")
        self.assertGreater(checked, 20, "no characteristic polynomials were sampled")

    def test_no_python_power_syntax_anywhere_in_the_derivation(self):
        """No `**` and no bare `lambda` anywhere in the scratchpad."""
        for p in _eigenvalues("eigenvalue") + _eigenvalues("eigenvalue_numeric"):
            self.assertNotIn("**", p.scratchpad,
                             f"the derivation uses Python's power syntax: "
                             f"{p.scratchpad[:160]!r}")
            for m in re.finditer(r"(?<![\\a-z])lambda(?![a-z])", p.scratchpad):
                self.fail(f"the derivation uses a bare 'lambda': "
                          f"{p.scratchpad[:160]!r}")

    def test_the_roots_are_found_not_asserted(self):
        """The factoring must be shown.

        A line that says "the roots of p are -5, 6, -8" with nothing
        between the polynomial and the roots teaches the model to state
        a set of numbers it cannot compute.
        """
        checked = 0
        for p in _eigenvalues():
            if "the eigenvalues of A are" not in p.scratchpad:
                continue
            checked += 1
            # the line must show the FACTORING, not just the roots
            self.assertIn(
                "(λ", p.scratchpad,
                f"the roots are asserted with no factoring shown: "
                f"{p.scratchpad[:260]!r}")
            self.assertIn(
                "p(", p.scratchpad,
                f"the polynomial is never evaluated, so the root is "
                f"asserted: {p.scratchpad[:260]!r}")
        self.assertGreater(checked, 20, "no eigenvalue derivations were sampled")


if __name__ == "__main__":
    unittest.main()
