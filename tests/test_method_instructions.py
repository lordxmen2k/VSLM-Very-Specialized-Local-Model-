"""The corpus must teach the METHOD the benchmark asks for.

The released LinAlg-Bench 3x3 problems do not all carry the generic
"Show all intermediate steps" closing. Many carry a TASK-SPECIFIC
instruction that names the method:

    determinant    "Use cofactor expansion along the first row."
    rank          "Use row reduction to find the number of non-zero rows
                    in row echelon form. Show your row operations."
    nullity       "Recall: nullity = number of columns - rank. First find
                    the rank using row reduction, then compute the nullity."
    eigenvalue    "Compute the characteristic polynomial det(A - lI) = 0
                    and solve for l. Show your steps. List all eigenvalues
                    inside \boxed{}."
    transpose     "The transpose A^T is obtained by swapping rows and
                    columns. Write out A^T explicitly."
    trace         "The trace is the sum of the diagonal elements."
    matrix_power  "Multiply A by itself. Show all row-by-column steps."
    matrix_vector "Compute each entry of b by taking the dot product of
                    each row of A with x."
    multiplication "Show your row-by-column computation steps clearly."

The corpus never mentioned any of them. The model was taught one method per
family and was then asked, in the benchmark's words, to use a different one.
For the determinant that is the difference between "expand along the first
row" and "eliminate below the pivot" - and the fine-tune scored 0.7% on
determinants while the base model, which was never taught either, scored
36%.

The corpus has to teach the method the question names.
"""
import csv
import os
import random
import re
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(REPO, "tests", "_cache")

sys.path.insert(0, os.path.join(REPO, "src"))

from la.problems import generate


def _benchmark_rows():
    rows = []
    for size in ("3x3", "4x4", "5x5"):
        path = os.path.join(CACHE, f"linalg_bench_{size}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append((size, row))
    return rows


def _method_instruction(text: str) -> str:
    """The method instruction the benchmark's own problem carries.

    The generic closing line is not a method; the specific sentence that
    says HOW to do the problem is.
    """
    idx = text.rfind("end{bmatrix}")
    if idx < 0:
        idx = text.rfind("end{pmatrix}")
    if idx < 0:
        return ""
    tail = " ".join(text[idx:].split())
    tail = tail.replace("end{bmatrix}", "").replace("end{pmatrix}", "").strip()
    if tail.startswith("Show all intermediate steps"):
        return ""
    return tail


class TestTheBenchmarkNamesAMethod(unittest.TestCase):

    def test_the_benchmark_really_does_name_methods(self):
        """If this stops being true, the fix below stops being needed."""
        rows = _benchmark_rows()
        if not rows:
            self.skipTest("benchmark files not fetched")
        named = [(_s, r) for _s, r in rows
                 if _method_instruction(r["problem_latex"])]
        self.assertGreater(len(named), 100,
                           "the released benchmark no longer names methods")
        subs = {r["Subcat"] for _s, r in named}
        self.assertIn("determinant", subs)
        self.assertIn("rank", subs)
        self.assertIn("nullity", subs)


class TestTheCorpusTeachesTheNamedMethod(unittest.TestCase):

    def _instruction_for(self, family: str, size: str = "3x3"):
        rows = _benchmark_rows()
        for _s, r in rows:
            if r["Subcat"] != _subcat_for(family):
                continue
            instr = _method_instruction(r["problem_latex"])
            if instr:
                return instr
        self.skipTest("benchmark files not fetched")

    def test_the_determinant_corpus_answers_the_cofactor_instruction(self):
        """The benchmark says 'use cofactor expansion along the first row'.

        The corpus must be able to produce that derivation, not only the
        elimination one.
        """
        instr = self._instruction_for("determinant")
        self.assertIn("cofactor", instr.lower())
        rng = random.Random(31)
        saw_cofactor = False
        # a 3x3 determinant is one size out of eight, so 60 draws are not
        # enough to see one
        for _ in range(400):
            p = generate("determinant", rng)
            if p is None or "3x3" not in p.question:
                continue
            if "cofactor" in p.scratchpad.lower():
                saw_cofactor = True
                break
        self.assertTrue(saw_cofactor,
                        "the corpus cannot produce a cofactor derivation")

    def test_the_rank_corpus_shows_row_operations(self):
        """The benchmark says 'use row reduction ... show your row
        operations'. The corpus's rank derivation must show them."""
        instr = self._instruction_for("rank")
        self.assertIn("row", instr.lower())
        rng = random.Random(32)
        p = generate("rank", rng)
        self.assertIsNotNone(p)
        self.assertRegex(p.scratchpad, r"R[123] <-")

    def test_the_nullity_corpus_shows_row_operations(self):
        instr = self._instruction_for("nullity")
        self.assertIn("row", instr.lower())
        rng = random.Random(33)
        p = generate("nullity", rng)
        self.assertIsNotNone(p)
        self.assertRegex(p.scratchpad, r"R[123] <-")

    def test_the_eigenvalue_corpus_computes_the_characteristic_polynomial(self):
        """The benchmark says 'compute the characteristic polynomial
        det(A - lI) = 0 and solve for l'."""
        instr = self._instruction_for("eigenvalue")
        self.assertIn("characteristic polynomial", instr.lower())
        rng = random.Random(34)
        for fam in ("eigenvalue", "eigenvalue_numeric"):
            p = generate(fam, rng)
            self.assertIsNotNone(p)
            self.assertIn("characteristic polynomial", p.scratchpad.lower(),
                          f"{fam} does not compute the characteristic polynomial")


def _subcat_for(family: str) -> str:
    from la.benchprompt import FAMILY_TO_SUBCAT
    return FAMILY_TO_SUBCAT[family]


if __name__ == "__main__":
    unittest.main()
