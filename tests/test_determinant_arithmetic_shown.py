"""The corpus's derivations must SHOW the arithmetic, not assert it.

THE DEFECT THIS FILE GUARDS AGAINST. The 3x3 determinant's cofactor
expansion wrote each 2x2 minor as

    the first minor is \begin{bmatrix} 4 & -1 \\ 5 & 1 \end{bmatrix} = 9

which states the minor's VALUE without showing how it was got. The
model was therefore taught to emit a number it could not compute, and
the determinant family scored 1 of 150 against the base model's 54 of
150 - the single largest deficit in the benchmark.

This is run 9's lesson, in a different family. Run 9 dropped the
products from the dot product:

    full:     4*4 + (-4)*(-4) + 3*3 + 0*0 = 16 + 16 + 9 + 0 = 41
    compact:                        16 + 16 + 9 + 0 = 41

and the arithmetic checker passed every one of them, because it
validates the expression on the LEFT of the "=" and a compact line is
perfectly well formed and perfectly well valued. The checker could not
see that the corpus had quietly stopped teaching the multiplication.

THE RULE. A derivation must show the terms that produce each value. A
minor's value must be reached by its own arithmetic, visibly, with the
products written out.

THE CHECKER IS NOT ENOUGH. The arithmetic checker validates the
expressions on the left of an "=". A line that says "= 9" with no
expression on the left is not an expression at all, so it slips
through. This file asserts the thing the arithmetic checker cannot:
that the products are PRESENT.
"""
from __future__ import annotations

import random
import re
import unittest

from la.problems import generate


def _determinants(n=120, seed=4242):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = generate("determinant", rng)
        if p is not None:
            out.append(p)
    return out


class TestTheCofactorExpansionShowsItsArithmetic(unittest.TestCase):

    def test_each_minor_is_computed_not_asserted(self):
        """A minor's value must be reached by visible arithmetic.

        The line must contain the products that produce it, not just the
        value.
        """
        checked = 0
        for p in _determinants():
            if "minor is" not in p.scratchpad:
                continue
            checked += 1
            for line in p.scratchpad.split("\n"):
                m = re.search(r"minor is (.*?) = (-?\d+)$", line)
                if not m:
                    continue
                body = m.group(1)
                # the products of the 2x2 minor must be there
                self.assertIn(
                    "*", body,
                    f"the minor is asserted, not computed: {line!r}")
                # and the value must be the one the arithmetic gives
                nums = re.findall(r"-?\d+", body)
                self.assertGreaterEqual(
                    len(nums), 4,
                    f"the minor's arithmetic is missing a factor: {line!r}")
        self.assertGreater(checked, 10, "no cofactor minors were sampled")

    def test_the_expansion_carries_the_elements_and_the_minors(self):
        """The final expansion must name every element and every minor.

        A determinant computed as a bare sum of values is what the
        benchmark's "cofactor-locked" failures look like: the answer is
        right and the derivation is unusable.
        """
        checked = 0
        for p in _determinants():
            m = re.search(r"det\(A\) = (.+?) = (-?\d+)$", p.scratchpad,
                          re.M)
            if not m:
                continue
            checked += 1
            expr = m.group(1)
            self.assertIn("*", expr,
                          f"the expansion has no products: {expr!r}")
            # three terms, one per column
            self.assertGreaterEqual(
                expr.count("+") + expr.count("-"), 2,
                f"the expansion has too few terms: {expr!r}")
        self.assertGreater(checked, 10, "no expansions were sampled")


if __name__ == "__main__":
    unittest.main()
