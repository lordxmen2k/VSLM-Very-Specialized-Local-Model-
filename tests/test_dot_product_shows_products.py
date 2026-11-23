"""The dot product must show its products, at EVERY size.

THE DEFECT THIS FILE GUARDS AGAINST. At 5x5 the corpus used a COMPACT
dot product that dropped the product terms:

    full:     4*4 + (-4)*(-4) + 3*3 + 0*0 = 16 + 16 + 9 + 0 = 41
    compact:                        16 + 16 + 9 + 0 = 41

The model was shown the partial sums and never the multiplication that
produced each term, so it learned to emit a number it could not
compute. `matrix_power` fell from 41 of 60 to 15 of 60 - more problems
than the entire regression - and `matrix_vector` fell to 8.

**THE ARITHMETIC CHECKER COULD NOT SEE IT.** It validates the
expression on the LEFT of each "=", and a compact line is perfectly
well formed and perfectly well valued. 3,800 problems, every expression
correct, and the corpus quietly teaching less.

A checker that asks "is what is present correct?" cannot see what
stopped being present. This file asserts that the products are PRESENT.

**THE 5x5 CAP IS NO LONGER A REASON.** The compact form was kept at
5x5 because the full form was measured at 1,587 tokens against a 1,536
cap. That measurement was taken against a longer derivation that no
longer exists, and the derivation has since been shortened. Measured
now, with the full form at every size:

    matrix_multiply 5x5   max 829 tokens
    matrix_power    5x5   max 784 tokens
    the cap                1536 tokens

Both fit with room to spare, so there is no reason to drop the products
at any size. The rule that follows from run 9 still stands: WHEN A
CHANGE IS MADE TO FIT A LIMIT, MEASURE THE LIMIT AT EVERY SIZE THE
CHANGE TOUCHES - and the limit has moved, so the change is no longer
needed.
"""
from __future__ import annotations

import random
import re
import unittest

from la.benchprompt import MAX_SEQ_LEN
from la.problems import generate

FAMILIES = ("matrix_multiply", "matrix_power", "matrix_vector")


def _examples(family, n=60, seed=4242):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = generate(family, rng)
        if p is not None:
            out.append(p)
    return out


def _entries_in(line):
    """The dot products in one line, as "products = value" strings.

    At the large sizes a line carries five entries - "STEP 2 row 1:
    (1,1) = ...; (1,2) = ...; ..." - and each has to be checked on its
    own. Splitting on the " = " that follows an entry label does it.
    """
    parts = re.split(r"\(\d+,\d+\) = |\([A-Z]\^\d\)_\{\d+\} = |"
                     r"\([A-Z]+\)_\{\d+\} = ", line)
    out = []
    for part in parts:
        part = part.strip()
        if not part or part.startswith("STEP") or part.startswith("GIVEN"):
            continue
        if "=" not in part:
            continue
        # the part is "products = value" or "products = sums = value"
        out.append(part)
    return out


class TestEveryDotProductShowsItsProducts(unittest.TestCase):

    def test_every_dot_product_shows_its_products(self):
        """A dot product line must contain the multiplication.

        The products are the part that costs the tokens and the part the
        model needs: without them it is asked to emit a number it cannot
        compute.
        """
        checked = 0
        for family in FAMILIES:
            for p in _examples(family):
                for line in p.scratchpad.split("\n"):
                    # an entry is "<label> = <products> = <value>", and at
                    # the large sizes several entries share a line, so the
                    # line is split on the entry labels first
                    for entry in _entries_in(line):
                        terms = entry
                        if "+" not in terms and "-" not in terms:
                            continue
                        checked += 1
                        self.assertIn(
                            "*", terms,
                            f"the dot product has no products: {entry!r}")
        self.assertGreater(checked, 30, "no dot products were sampled")

    def test_the_full_form_fits_the_cap_at_every_size(self):
        """The compact form is no longer needed at any size.

        The measurement that kept it at 5x5 - 1,587 tokens against a
        1,536 cap - was taken against a longer derivation that no
        longer exists. The full form now measures 829 tokens at 5x5 for
        the multiplication and 784 for the power.
        """
        import la.problems as P

        original = P._dot_product_compact
        try:
            P._dot_product_compact = P._dot_product
            for family in ("matrix_multiply", "matrix_power"):
                worst = 0
                for p in _examples(family, 120):
                    m = re.search(r"\\begin\{[bp]matrix\}(.*?)\\end\{[bp]matrix\}",
                                  p.question, re.S)
                    if not m:
                        continue
                    rows = [ln for ln in m.group(1).split("\\\\") if ln.strip()]
                    if not rows:
                        continue
                    tokens = (len(p.question.split())
                              + len(p.scratchpad.split()))
                    if len(rows) == 5:
                        worst = max(worst, tokens)
                self.assertLessEqual(
                    worst, MAX_SEQ_LEN,
                    f"{family} at 5x5 with the full form is {worst} tokens, "
                    f"over the {MAX_SEQ_LEN} cap, so the products cannot be "
                    f"shown at that size")
        finally:
            P._dot_product_compact = original

    def test_the_corpus_uses_the_full_form_at_every_size(self):
        """And the corpus actually uses it, at 5x5 too."""
        for family in ("matrix_multiply", "matrix_power"):
            checked = 0
            for p in _examples(family, 120):
                m = re.search(r"\\begin\{[bp]matrix\}(.*?)\\end\{[bp]matrix\}",
                              p.question, re.S)
                if not m:
                    continue
                rows = [ln for ln in m.group(1).split("\\\\") if ln.strip()]
                if len(rows) != 5:
                    continue
                checked += 1
                self.assertIn(
                    "*", p.scratchpad,
                    f"{family} at 5x5 drops the products: "
                    f"{p.scratchpad[:200]!r}")
            self.assertGreater(checked, 5, "no 5x5 examples were sampled")


if __name__ == "__main__":
    unittest.main()
