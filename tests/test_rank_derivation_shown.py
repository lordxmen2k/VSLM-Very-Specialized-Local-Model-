"""The rank derivation must show its row operations, and the rank must be
read off the echelon form.

THE DEFECT THIS FILE GUARDS AGAINST. The reduction used to SCALE the
pivot rows to 1:

    R1 <- R1 / (-14)
    the echelon form is \\begin{pmatrix} 1 & \\frac{1}{14} & \\frac{2}{7}
                       \\ 0 & 0 & 0 \\\\ 0 & 0 & 0 \\end{pmatrix}

and the row operations were STATED WITHOUT THEIR RESULTS, so the
echelon form's entries appeared from nowhere.

**A NOTE ON FRACTIONS.** An earlier version of this file forbade
fractions in the reduction. That was wrong, and the measurement says
so: the BENCHMARK's own rank and nullity matrices, reduced the ordinary
way, produce 2,000 fractional entries across the 120 of them. The
fractional reduction is the natural route for those matrices; what the
benchmark never does is give a fractional ANSWER - all 150 of its
determinant answers and all 120 of its rank and nullity answers are
integers.

So the rule is the narrower and correct one: the reduction may carry
fractions, but the RANK is the number of NON-ZERO ROWS in the echelon
form, and the derivation must show each operation together with the row
it produced.

The pivot rows are not scaled to 1 any more. Scaling divides a whole
row by the pivot and manufactures fractions the reduction did not need,
and the rank does not need it - the benchmark asks for "the number of
non-zero rows in row echelon form".
"""
from __future__ import annotations

import random
import re
import unittest

from la.problems import generate


def _rank_problems(family="rank", n=90, seed=4242):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = generate(family, rng)
        if p is not None:
            out.append(p)
    return out


class TestTheRankReductionIsShown(unittest.TestCase):

    def test_each_row_operation_shows_the_row_it_produced(self):
        """An operation with no result teaches the model to assert.

        The operation must be followed by the row it produced, so the
        model can see the arithmetic.
        """
        checked = 0
        for family in ("rank", "nullity"):
            for p in _rank_problems(family):
                lines = p.scratchpad.split("\n")
                for i, line in enumerate(lines):
                    if not re.search(r"R\d+ <- ", line):
                        continue
                    checked += 1
                    nxt = lines[i + 1] if i + 1 < len(lines) else ""
                    self.assertRegex(
                        nxt, r"\d",
                        f"the row operation {line!r} is not followed by the "
                        f"row it produced")
        self.assertGreater(checked, 50, "no row operations were sampled")

    def test_the_echelon_form_is_never_scaled_to_ones_by_dividing(self):
        """The pivot rows are not scaled.

        "R1 <- R1 / (16)" divides a whole row and manufactures the
        fractions the reduction did not need. The rank is the number of
        non-zero rows, so the scaling buys nothing.
        """
        checked = 0
        for family in ("rank", "nullity"):
            for p in _rank_problems(family):
                checked += 1
                self.assertNotRegex(
                    p.scratchpad, r"R\d+ <- R\d+ / \(",
                    "the reduction scales a pivot row by dividing it: "
                    f"{p.scratchpad[:200]!r}")
        self.assertGreater(checked, 50, "no rank problems were sampled")

    def test_the_rank_is_read_off_the_echelon_form(self):
        """The derivation must count the non-zero rows.

        The benchmark asks for "the number of non-zero rows in row echelon
        form", so the derivation has to show that count.
        """
        checked = 0
        for family in ("rank", "nullity"):
            for p in _rank_problems(family):
                if "pivot rows" not in p.scratchpad:
                    continue
                checked += 1
                m = re.search(r"has (\d+) pivot rows", p.scratchpad)
                self.assertIsNotNone(m)
                if family == "rank":
                    self.assertEqual(
                        m.group(1), p.answer,
                        "the pivot count and the answer disagree")
        self.assertGreater(checked, 50, "no rank derivations were sampled")


if __name__ == "__main__":
    unittest.main()
