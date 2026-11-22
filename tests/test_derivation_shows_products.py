"""The derivation must SHOW the multiplications, not just their results.

RUN 9 SCORED 225/660 = 34.1%, down from the 259-run's 39.2% and run 7's
42.7%.

The cause was a change run 9 made to fit the 5x5 examples under the
1536-token sequence cap. The full dot product is:

    4*4 + (-4)*(-4) + 3*3 + 0*0 = 16 + 16 + 9 + 0 = 41

and the compact form run 9 used at the large sizes is:

    = 16 + 16 + 9 + 0 = 41

The compact form drops the PRODUCTS. The model is shown the partial sums
but never the multiplication that produced each one, so it is taught to
emit a sum whose terms it has no idea how to compute.

The cost was immediate and confined to the families that use the dot
product:

    family            259-run   run 9   base
    matrix_power         41      15      34     -26
    matrix_vector        18       8      44     -10
    multiplication       33      29      25      -4
    determinant           4       3      54      -1

`matrix_power` lost 26 problems - more than the whole regression - and
the 4x4 column fell from 95 to 70.

The corpus's own arithmetic checker did NOT catch this, and that is the
point of this module: the checker validates the expression on the LEFT of
each "=", so a line whose left side is already a sum of products is
perfectly well formed and perfectly well valued. Nothing flagged that the
model was being taught less.

So the property is pinned here directly: at every size the corpus can
afford, the derivation must show the product terms.
"""
import os
import random
import re
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, os.path.join(REPO, "src"))

from la.problems import generate


def _dot_product_lines(scratchpad):
    """The derivation lines that state a dot product.

    A dot product line names an entry and gives it a value:

        STEP 2 row 1: (A^2)_{11} = 4 - 4 - 3 + 0 = -3; (A^2)_{12} = ...

    The line is recognised by the ENTRY it names - "(A^2)_{ij}" or
    "(AB)_{ij}" - not by its arithmetic, because the arithmetic is
    exactly what differs between the two forms.
    """
    out = []
    for line in scratchpad.splitlines():
        if re.search(r"\((?:A\^2|AB|AB|A)\)\{?\d", line):
            out.append(line)
        elif re.search(r"\(A\^2\)_\{?\d", line) or re.search(r"\(AB\)_\{?\d", line):
            out.append(line)
    return out


class TestTheDerivationShowsTheMultiplications(unittest.TestCase):
    """The products must be visible wherever there is room for them."""

    def test_matrix_power_shows_its_product_terms(self):
        """At 3x3 and 4x4 the full form fits easily - 591 and 983
        tokens against a 1536 cap - so there is no excuse for dropping
        the products."""
        for size in (3, 4):
            with self.subTest(size=f"{size}x{size}"):
                self._assert_products_shown("matrix_power", size)

    def test_matrix_multiply_shows_its_product_terms(self):
        for size in (3, 4):
            with self.subTest(size=f"{size}x{size}"):
                self._assert_products_shown("matrix_multiply", size)

    def _assert_products_shown(self, family, size):
        rng = random.Random(4242)
        seen = 0
        for _ in range(200):
            p = generate(family, rng)
            if p is None:
                continue
            if f"{size}x{size}" not in p.question:
                continue
            seen += 1
            lines = _dot_product_lines(p.scratchpad)
            if not lines:
                continue
            for line in lines:
                self.assertIn(
                    "*", line,
                    f"{family} at {size}x{size}: the derivation states "
                    f"{line!r} with no product terms. The model is shown "
                    f"the sum but not the multiplication that produced "
                    f"each term, so it is taught to emit a number it "
                    f"cannot compute.")
            if seen >= 3:
                break
        if not seen:
            self.skipTest(f"no {size}x{size} {family} problem was generated")

    def test_the_5x5_form_still_fits_the_sequence_cap(self):
        """The 5x5 example DOES need a reduced form - the full one runs
        to 1,601 tokens against a 1,536 cap, and a truncated example
        loses the end of its completion, which is where the plot spec
        is. So 5x5 is allowed to drop the products, and everything else
        is not.

        This test exists so the 5x5 exception stays an EXCEPTION: if a
        future change makes the 5x5 form fit, this test will say so and
        the products can go back.
        """
        from la.benchprompt import MAX_SEQ_LEN as cap
        rng = random.Random(4242)
        worst = 0
        for _ in range(300):
            p = generate("matrix_power", rng)
            if p is None:
                continue
            worst = max(worst, len(p.question) + len(p.completion()))
        self.assertLessEqual(
            worst / 2.04, cap,
            f"the longest matrix_power example is about {worst / 2.04:.0f} "
            f"tokens, over the {cap} cap")


class TestTheArithmeticCheckerWouldNotHaveCaughtIt(unittest.TestCase):
    """A demonstration, so the failure mode is not forgotten.

    The corpus's arithmetic checker validates the expression on the LEFT
    of each "=". A compact dot product line is
    "= 16 + 16 + 9 + 0 = 41" - the left side is a valid arithmetic
    expression and it is valued correctly. The checker passes it.

    So the checker cannot see a derivation that has quietly stopped
    teaching something. The contract has to be pinned by a test that
    looks at what is PRESENT, not at whether what is present is right.
    """

    def test_a_compact_line_is_well_formed_and_well_valued(self):
        from la.check_arithmetic import check_problem
        from la.problems import Problem

        p = Problem(
            "matrix_power",
            "Compute A^2 for the 3x3 matrix A.",
            "GIVEN: A, k = 2\n"
            "STEP 1 (A^2)_{11} = 16 + 16 + 9 + 0 = 41",
            "41", "41", {"kind": "matrix_heatmap", "data": {}, "labels": {},
                         "annotate": []}, 1, ["exact", "scalar"])
        self.assertEqual(
            [], check_problem(p),
            "the arithmetic checker accepted a line with no product terms, "
            "which is exactly why this needs a separate test")


if __name__ == "__main__":
    unittest.main()
