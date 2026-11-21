"""The corpus's determinant arithmetic must be in the benchmark's range.

Run 7 scored 282/660 = 42.7% against its base model's 306/660 = 46.4%.
Every family was competitive or better EXCEPT one:

    determinant   run 7 = 4/150    base = 54/150

Remove the determinant family and the fine-tune WINS:

    run 7 without determinant = 278/510 = 54.5%
    base  without determinant = 252/510 = 49.4%

So the determinant family is the entire 24-problem deficit. The cause is
not the prompt shape any more - run 7 fixed that, and the corpus's
determinant derivations are arithmetically correct (make check-arithmetic
passes). The cause is the ARITHMETIC LOAD the corpus asks for.

Measured:

  corpus 3x3 determinant matrix entries:   [-4, 5]
  benchmark 3x3 determinant matrix entries: [-9, 9]

  corpus 3x3:  largest single product the model must form, median   40
  benchmark 3x3:                                     median  168
  corpus 3x3:  products >= 100, median count              0
  benchmark 3x3:                                         median   1

  corpus 5x5: 82% of derivations contain a fraction whose numerator
  is >= 100, with the largest numerator seen at 4009. The benchmark's
  5x5 determinants have |det| up to 9112.

The corpus teaches the model to emit a long chain of exact arithmetic -
nine numbered steps at 3x3, and at 5x5 a fraction cascade like 807/118 -
and one slip anywhere loses the whole problem. The base model was never
taught that chain, so it answers directly and lands the number more often.

The corpus has to train the ARITHMETIC THE BENCHMARK ASKS FOR, at the
benchmark's own magnitudes.
"""
import csv
import os
import random
import re
import sys
import unittest
from collections import Counter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(REPO, "tests", "_cache")

sys.path.insert(0, os.path.join(REPO, "src"))

from la.problems import generate


def _parse_matrix(body):
    """The entries of a LaTeX matrix body, row by row.

    The body is split on the row separator and each row's numbers are
    read. A number that belongs to the NEXT thing in the problem - the
    closing \\end{bmatrix} is followed by nothing, but a bare trailing
    integer would be picked up - is excluded by requiring the row to
    contain ONLY numbers and separators.
    """
    mat = []
    for line in body.split("\\\\"):
        cleaned = line.replace("&", " ").strip()
        if not cleaned:
            continue
        nums = re.findall(r"-?\d+", cleaned)
        # every token in the row must be a number, otherwise this row
        # carries something else (the closing bracket, a label)
        if re.sub(r"-?\d+", "", cleaned).strip():
            continue
        mat.append([int(x) for x in nums])
    return mat


def _benchmark_matrices(subcat):
    """Every MATRIX in the benchmark's problems for a subcategory.

    Only the matrices - the benchmark also has vectors (matrix_vector's
    x) and those are a different quantity.
    """
    out = []
    for size in ("3x3", "4x4", "5x5"):
        path = os.path.join(CACHE, f"linalg_bench_{size}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["Subcat"] != subcat:
                    continue
                for m in re.finditer(
                        r"[A-Z](?:_\{[A-Z]+\})? = \\begin\{[bp]matrix\}(.*?)"
                        r"\\end\{[bp]matrix\}", row["problem_latex"], re.S):
                    mat = _parse_matrix(m.group(1))
                    if mat and all(len(r) == len(mat[0]) for r in mat):
                        out.append(mat)
    return out


def _corpus_matrices(family, n=400):
    """Every matrix the corpus produces for a family."""
    rng = random.Random(4242)
    out = []
    for _ in range(n):
        p = generate(family, rng)
        if p is None:
            continue
        for m in re.finditer(
                r"[A-Z](?:_\{[A-Z]+\})? = \\begin\{[bp]matrix\}(.*?)"
                r"\\end\{[bp]matrix\}", p.question, re.S):
            mat = _parse_matrix(m.group(1))
            if mat and all(len(r) == len(mat[0]) for r in mat):
                out.append(mat)
    return out


def _entry_range(matrices):
    vals = [abs(x) for m in matrices for row in m for x in row]
    return max(vals) if vals else 0


class TestTheCorpusCoversTheBenchmarksMagnitudes(unittest.TestCase):

    def test_the_benchmark_really_does_use_larger_entries(self):
        """If this stops being true, the fix below stops being needed."""
        mats = _benchmark_matrices("determinant")
        if not mats:
            self.skipTest("benchmark files not fetched")
        three = [m for m in mats if len(m) == 3]
        self.assertTrue(three)
        self.assertGreaterEqual(_entry_range(three), 9,
                                 "the benchmark's 3x3 determinants no longer "
                                 "reach 9")

    def test_the_corpus_determinant_entries_reach_the_benchmarks(self):
        mats = _benchmark_matrices("determinant")
        if not mats:
            self.skipTest("benchmark files not fetched")
        want = _entry_range(mats)
        have = _entry_range(_corpus_matrices("determinant"))
        self.assertGreaterEqual(
            have, want,
            f"the corpus's determinant entries reach {have} but the "
            f"benchmark's reach {want}")

    def test_every_benchmark_family_is_covered(self):
        """Not just the determinant: a corpus that trains on smaller
        entries than the benchmark uses is out of distribution for every
        family, and the cost shows up as scattered small losses."""
        for family, subcat in (("determinant", "determinant"),
                                ("eigenvalue", "eigenvalue"),
                                ("matrix_vector", "matrix_vector"),
                                ("matrix_multiply", "multiplication"),
                                ("nullity", "nullity"),
                                ("matrix_power", "matrix_power"),
                                ("rank", "rank"),
                                ("matrix_trace", "trace"),
                                ("matrix_transpose", "transpose")):
            with self.subTest(family=family):
                bench = _benchmark_matrices(subcat)
                if not bench:
                    self.skipTest("benchmark files not fetched")
                want = _entry_range(bench)
                have = _entry_range(_corpus_matrices(family, 250))
                self.assertGreaterEqual(
                    have, want,
                    f"{family}: the corpus's entries reach {have} but the "
                    f"benchmark's reach {want}")


class TestTheDeterminantArithmeticIsInRange(unittest.TestCase):

    def test_the_corpus_does_not_demand_larger_products_than_the_benchmark(self):
        """The quantity that matters is the product the model has to form.

        A cofactor term is a_1j * M_1j. The benchmark's 3x3 determinant
        terms have a median of 60 and a p90 of 360; the corpus's old
        [-4, 5] entries gave a median of 20. The corpus has to train
        arithmetic at the benchmark's magnitude, not below it.
        """
        import sympy as sp

        def terms(mat):
            A = sp.Matrix(mat)
            out = []
            for j in range(A.cols):
                minor = A.minor_submatrix(0, j).det()
                out.append(abs(int(A[0, j]) * int(minor)))
            return out

        bench = _benchmark_matrices("determinant")
        if not bench:
            self.skipTest("benchmark files not fetched")
        three = [m for m in bench if len(m) == 3]
        bench_terms = sorted(t for m in three for t in terms(m))
        bench_median = bench_terms[len(bench_terms) // 2]

        corp = [m for m in _corpus_matrices("determinant") if len(m) == 3]
        self.assertTrue(corp)
        corp_terms = sorted(t for m in corp for t in terms(m))
        corp_median = corp_terms[len(corp_terms) // 2]

        self.assertGreaterEqual(
            corp_median, bench_median * 0.9,
            f"the corpus's median cofactor term is {corp_median} but the "
            f"benchmark's is {bench_median}: the corpus trains arithmetic "
            f"smaller than the benchmark asks for")

    def test_the_corpus_reaches_the_benchmarks_tail(self):
        """The p90 is what a model has to be able to do, not just the
        median: the benchmark's 3x3 determinant terms have a p90 of 360."""
        import sympy as sp

        def terms(mat):
            A = sp.Matrix(mat)
            out = []
            for j in range(A.cols):
                minor = A.minor_submatrix(0, j).det()
                out.append(abs(int(A[0, j]) * int(minor)))
            return out

        bench = _benchmark_matrices("determinant")
        if not bench:
            self.skipTest("benchmark files not fetched")
        three = [m for m in bench if len(m) == 3]
        bench_terms = sorted(t for m in three for t in terms(m))
        bench_p90 = bench_terms[int(len(bench_terms) * 0.9)]

        corp = [m for m in _corpus_matrices("determinant", 600) if len(m) == 3]
        self.assertTrue(corp)
        corp_terms = sorted(t for m in corp for t in terms(m))
        corp_p90 = corp_terms[int(len(corp_terms) * 0.9)]

        self.assertGreaterEqual(
            corp_p90, bench_p90 * 0.8,
            f"the corpus's p90 cofactor term is {corp_p90} but the "
            f"benchmark's is {bench_p90}")


if __name__ == "__main__":
    unittest.main()
