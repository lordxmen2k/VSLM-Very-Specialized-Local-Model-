"""Every printed derivation must add up.

Two failure modes this catches:

  * a scratchpad states an expression and then a value that the expression
    does not evaluate to - which teaches the model to fabricate. The ground
    truth is computed by sympy and is always right, so this is invisible to
    a test that only compares the answer with the truth.
  * the step numbers jump, so the model learns a derivation with holes in it.

Both are properties of the printed text, so both have to be read off the text.
"""
import random
import re
import unittest

from la.check_arithmetic import check_problem
from la.problems import GENERATORS

SAMPLES = 30


class TestScratchpadArithmetic(unittest.TestCase):
    def test_every_family_prints_arithmetic_that_adds_up(self):
        rng = random.Random(2024)
        complaints = []
        for family, gen in sorted(GENERATORS.items()):
            seen = 0
            tries = 0
            while seen < SAMPLES and tries < SAMPLES * 60:
                tries += 1
                p = gen(rng)
                if p is None:
                    continue
                seen += 1
                complaints.extend(check_problem(p))
        self.assertEqual(
            [], complaints,
            "printed derivations that do not evaluate to their own answer:\n"
            + "\n".join("  " + c for c in complaints[:20]))

    def test_step_numbers_are_contiguous(self):
        # "STEP 3b" is a deliberate sub-step of STEP 3, used by the
        # orthonormalisation family. A bare repeat of a number is a bug.
        pattern = re.compile(r"^STEP (\d+)([a-z]?)\b", re.M)
        rng = random.Random(99)
        bad = []
        for family, gen in sorted(GENERATORS.items()):
            seen = 0
            tries = 0
            while seen < SAMPLES and tries < SAMPLES * 60:
                tries += 1
                p = gen(rng)
                if p is None:
                    continue
                seen += 1
                marks = pattern.findall(p.scratchpad)
                if not marks:
                    continue
                nums = [int(n) for n, _ in marks]
                suffixes = [s for _, s in marks]
                distinct = []
                for n in nums:
                    if not distinct or distinct[-1] != n:
                        distinct.append(n)
                if distinct != list(range(1, len(distinct) + 1)):
                    bad.append(f"{family}: step numbers {nums}")
                for i in range(1, len(marks)):
                    if nums[i] == nums[i - 1]:
                        if not suffixes[i]:
                            bad.append(f"{family}: STEP {nums[i]} repeated "
                                       f"with no sub-step letter")
                        if suffixes[i - 1]:
                            bad.append(f"{family}: two sub-steps share "
                                       f"STEP {nums[i]}")
        self.assertEqual([], bad, "step numbering problems:\n" + "\n".join(bad[:20]))

    def test_determinant_expansion_includes_the_matrix_element(self):
        # the 3x3 cofactor expansion once dropped the a_1j factor, so the
        # printed sum was the signed minors rather than the determinant
        import sympy as sp
        rng = random.Random(7)
        seen = 0
        while seen < 40:
            p = GENERATORS["determinant"](rng)
            if p is None:
                continue
            line = p.scratchpad.splitlines()[1]
            if "expand along the first row" not in line:
                continue
            seen += 1
            expr = line.split("det(A) = ", 1)[1].rsplit(" = ", 1)[0]
            value = sp.nsimplify(sp.sympify(expr.replace(" ", "")))
            want = sp.nsimplify(sp.Rational(p.answer))
            self.assertEqual(0, sp.simplify(value - want),
                             f"expansion {expr} = {value}, answer {p.answer}")
        self.assertGreater(seen, 0, "no 3x3 determinant was generated")


if __name__ == "__main__":
    unittest.main()
