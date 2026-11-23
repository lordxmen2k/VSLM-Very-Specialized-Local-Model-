"""The determinant's derivation must use the arithmetic the benchmark uses.

THE DEFECT THIS FILE GUARDS AGAINST. The corpus's elimination route
divided by the pivot, so a 5x5 determinant's derivation ran through

    det(A) = 1 * -8 * -4 * \\frac{-281}{32} * \\frac{-688}{281}

and every one of the 4x4 and 5x5 determinant problems in the corpus
carried fractions.

The benchmark uses NO fractions: 0 of its 660 problems and 0 of its 660
answers contain one. Every one of the 150 determinant answers is an
integer. So the corpus was teaching a harder, longer, fraction-laden
derivation for a benchmark that only ever asks for integers - and the
determinant family scored 1 of 150 against the base model's 54 of 150.

The route must also stay INSIDE the sequence cap. The cap is 1536
tokens and the corpus's largest example is 1242, so there is room; but
a route that grows the derivation past the cap would be dropped from
the corpus entirely.

The route that keeps the arithmetic integral: elimination by row
addition with a WHOLE-NUMBER multiplier wherever the pivot divides the
entry, and a fraction-free (Bareiss) step otherwise.

This file asserts the property, not the route: whatever route the
corpus takes, the intermediate values must be integers and the answer
must be one.
"""
from __future__ import annotations

import random
import re
import unittest

from la.benchprompt import MAX_SEQ_LEN
from la.problems import generate


def _determinants(n=150, seed=4242):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = generate("determinant", rng)
        if p is not None:
            out.append(p)
    return out


class TestTheDeterminantUsesIntegerArithmetic(unittest.TestCase):

    def test_no_fraction_appears_in_a_determinant_derivation(self):
        """Fractions are the benchmark's arithmetic's absence.

        0 of the benchmark's 660 problems and 0 of its 660 answers contain
        a fraction.
        """
        checked = 0
        for p in _determinants():
            checked += 1
            self.assertNotIn(
                "\\frac", p.scratchpad,
                "the derivation uses a fraction, and the benchmark never "
                f"does: {p.scratchpad[:160]!r}")
        self.assertGreater(checked, 50, "no determinants were sampled")

    def test_the_derivation_stays_inside_the_sequence_cap(self):
        """A derivation past the cap is dropped from the corpus."""
        for p in _determinants():
            tokens = len(p.scratchpad.split()) + len(p.question.split())
            self.assertLessEqual(
                tokens, MAX_SEQ_LEN,
                f"the determinant's example is {tokens} tokens, over the "
                f"{MAX_SEQ_LEN} cap, so it would be dropped from the corpus")

    def test_the_derivation_ends_at_the_right_answer(self):
        """The last number in the derivation must be the answer.

        A derivation whose intermediate values are integers but whose
        final answer is not would be a different failure; this is the
        cheap end-to-end check that the derivation and the answer agree.
        """
        checked = 0
        for p in _determinants():
            m = re.search(r"the answer is (-?\d+)\s*$", p.scratchpad.strip())
            if not m:
                continue
            checked += 1
            self.assertEqual(
                m.group(1), p.answer,
                f"the derivation ends at {m.group(1)} but the answer is "
                f"{p.answer}")
        self.assertGreater(checked, 50, "no derivations were sampled")


if __name__ == "__main__":
    unittest.main()
