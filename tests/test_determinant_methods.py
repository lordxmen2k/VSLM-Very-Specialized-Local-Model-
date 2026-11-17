"""Derivations must perform their work, not assert its result."""
import random
import unittest

from la.problems import generate


class TestDeterminantMethods(unittest.TestCase):

    def test_3x3_determinants_have_both_cofactor_and_elimination_paths(self):
        rng = random.Random(7)
        methods = set()
        for _ in range(80):
            p = generate("determinant", rng)
            if p is None or "3x3" not in p.question:
                continue
            text = p.scratchpad.lower()
            if "cofactor" in text:
                methods.add("cofactor")
            if "row" in text and "eliminate" in text:
                methods.add("elimination")
        self.assertEqual(methods, {"cofactor", "elimination"})

    def test_elimination_derivation_shows_a_resulting_matrix(self):
        rng = random.Random(8)
        saw = False
        for _ in range(80):
            p = generate("determinant", rng)
            if p is None or "cofactor" in p.scratchpad.lower():
                continue
            saw = True
            self.assertIn("\\begin{", p.scratchpad)
            break
        self.assertTrue(saw, "no elimination determinant generated")


class TestRankDerivationsShowOperations(unittest.TestCase):

    def test_rank_derivation_shows_row_operations(self):
        p = generate("rank", random.Random(9))
        self.assertIsNotNone(p)
        self.assertRegex(p.scratchpad, r"R[123] <-")

    def test_nullity_derivation_shows_row_operations(self):
        p = generate("nullity", random.Random(9))
        self.assertIsNotNone(p)
        self.assertRegex(p.scratchpad, r"R[123] <-")


if __name__ == "__main__":
    unittest.main()
