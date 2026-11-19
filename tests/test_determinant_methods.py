"""Derivations must perform their work, not assert its result."""
import random
import re
import unittest

from la.problems import generate


class TestDeterminantMethods(unittest.TestCase):

    def test_3x3_determinants_have_both_cofactor_and_elimination_paths(self):
        """Both routes must be reachable.

        The corpus draws a 3x3 problem in one size out of eight, so the
        sample has to be large enough to see both. An earlier version of
        this test used 80 draws and saw 11 three-by-three problems, which
        is not enough to establish that a rare path is reachable at all.
        """
        rng = random.Random(7)
        methods = set()
        seen_3x3 = 0
        for _ in range(400):
            p = generate("determinant", rng)
            if p is None:
                continue
            m = re.search(r"of the (\d)x\1", p.question)
            if not m or m.group(1) != "3":
                continue
            seen_3x3 += 1
            text = p.scratchpad.lower()
            if "cofactor" in text:
                methods.add("cofactor")
            if "row" in text and "eliminate" in text:
                methods.add("elimination")
        self.assertGreaterEqual(seen_3x3, 20,
                                "not enough 3x3 determinants sampled")
        self.assertEqual(methods, {"cofactor", "elimination"})

    def test_cofactors_are_the_majority_route_at_3x3(self):
        """The benchmark instructs cofactors in 18 of its 50 three-by-three
        determinant problems, so cofactors must be the common route."""
        rng = random.Random(7)
        counts = {"cofactor": 0, "elimination": 0}
        for _ in range(400):
            p = generate("determinant", rng)
            if p is None:
                continue
            m = re.search(r"of the (\d)x\1", p.question)
            if not m or m.group(1) != "3":
                continue
            text = p.scratchpad.lower()
            if "cofactor" in text:
                counts["cofactor"] += 1
            elif "row" in text and "eliminate" in text:
                counts["elimination"] += 1
        self.assertGreater(counts["cofactor"], counts["elimination"],
                           "cofactors must be the common 3x3 route")

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
