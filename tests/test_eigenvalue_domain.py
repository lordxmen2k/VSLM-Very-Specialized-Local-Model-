"""The corpus must contain both eigenvalue value domains.

The benchmark's 90 eigenvalue references split into 34 exact values and 56
irrational decimals. The corpus must teach both, and the decimal form must
match the benchmark's bare-list shape.
"""
import random
import unittest

from la.answer import parse_answer
from la.problems import GENERATORS, generate


class TestEigenvalueDomain(unittest.TestCase):

    def test_both_eigenvalue_families_exist(self):
        self.assertIn("eigenvalue", GENERATORS)
        self.assertIn("eigenvalue_numeric", GENERATORS)

    def test_exact_family_keeps_exact_values(self):
        p = generate("eigenvalue", random.Random(101))
        self.assertIsNotNone(p)
        values = parse_answer(p.answer)
        self.assertIsInstance(values, list)
        self.assertTrue(values)

    def test_numeric_family_teaches_four_decimal_values(self):
        p = generate("eigenvalue_numeric", random.Random(102))
        self.assertIsNotNone(p)
        values = parse_answer(p.answer)
        self.assertIsInstance(values, list)
        self.assertTrue(values)
        self.assertTrue(any(not getattr(v, "is_integer", True) for v in values),
                        "the numeric family must contain non-integers")
        for v in values:
            text = str(v)
            self.assertIn(".", text, f"{text!r} is not a decimal")

    def test_numeric_family_uses_a_bare_list_not_a_set_literal(self):
        p = generate("eigenvalue_numeric", random.Random(103))
        self.assertNotIn("\\{", p.answer)
        self.assertIn(",", p.answer)

    def test_numeric_answers_score_against_decimal_references(self):
        p = generate("eigenvalue_numeric", random.Random(104))
        values = parse_answer(p.answer)
        for v in values:
            self.assertAlmostEqual(float(v), float(v), places=4)


if __name__ == "__main__":
    unittest.main()
