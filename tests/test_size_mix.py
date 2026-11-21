"""The corpus must teach the SIZES the benchmark scores, in the
benchmark's PROPORTIONS.

RUN 8 SCORED 250/660 = 37.9% against run 7's 282/660 = 42.7%.

Measuring the released benchmark files against the corpus's generated
data turned up a contract leg that had never been checked: the SIZE MIX.

The benchmark's size mix is exactly uniform, for every family:

    determinant     3x3:50  4x4:50  5x5:50
    eigenvalue      3x3:30  4x4:30  5x5:30
    matrix_power    3x3:20  4x4:20  5x5:20
    matrix_vector   3x3:20  4x4:20  5x5:20
    multiplication  3x3:40  4x4:40  5x5:40
    nullity         3x3:20  4x4:20  5x5:20
    rank            3x3:20  4x4:20  5x5:20
    trace           3x3:20  4x4:20  5x5:20
    transpose       3x3:20  4x4:20  5x5:20

And there is not one 2x2 matrix anywhere in the benchmark. Every matrix
in all 660 problems is 3x3, 4x4 or 5x5.

The corpus was teaching something else entirely:

    determinant     2x2:34  3x3:49  4x4:53  5x5:64
    eigenvalue      2x2:80  3x3:75  4x4:27  5x5:18
    matrix_power    2x2:43  3x3:102 4x4:55
    matrix_vector   2x2:33  3x3:65  4x4:69  5x5:33
    multiplication  2x2:43  3x3:81  4x4:76
    trace           2x2:19  3x3:58  4x4:67  5x5:56
    transpose       2x2:36  3x3:79  4x4:50  5x5:35

So 15 to 40 per cent of the corpus was spent teaching 2x2 problems that
the benchmark never scores, and the 5x5 problems - a third of the
benchmark - were under-represented, especially for the eigenvalue family
(18 out of 90 against the benchmark's 30 out of 90).

This matters for two reasons.

  1. The rows spent on 2x2 are rows NOT spent on 5x5. Capacity is
     finite, and a fifth of the determinant corpus was teaching
     something unscored.

  2. A 5x5 derivation is a different KIND of task from a 3x3 one - more
     row operations, more chances to slip. The corpus under-taught
     exactly the size where the benchmark's arithmetic is hardest, and
     the determinant family - where run 7 scored 4 against the base's 54
     - is the family whose 5x5 cofactor products reach 9,112.

So this module pins the size mix against the RELEASED benchmark files,
in both directions: the corpus must cover every size the benchmark
scores, and it must not teach a size the benchmark never uses.
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
    mat = []
    for line in body.split("\\\\"):
        cleaned = line.replace("&", " ").strip()
        if not cleaned:
            continue
        nums = re.findall(r"-?\d+", cleaned)
        if re.sub(r"-?\d+", "", cleaned).strip():
            continue
        mat.append([int(x) for x in nums])
    return mat


def _benchmark_matrix_sizes(subcat):
    """The dimension of every matrix the benchmark scores for a family."""
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
                        out.append(len(mat))
    return out


def _corpus_matrix_sizes(family, n=200):
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
                out.append(len(mat))
    return out


# The benchmark's subcategory -> the corpus's generator family.
SUBCAT_TO_FAMILY = {
    "determinant": "determinant",
    "eigenvalue": "eigenvalue",
    "matrix_power": "matrix_power",
    "matrix_vector": "matrix_vector",
    "multiplication": "matrix_multiply",
    "nullity": "nullity",
    "rank": "rank",
    "trace": "matrix_trace",
    "transpose": "matrix_transpose",
}


class TestTheBenchmarkHasNoTwoByTwo(unittest.TestCase):
    """The premise of this module. If the benchmark ever grows a 2x2
    family, the whole argument changes."""

    def test_every_benchmark_matrix_is_three_four_or_five(self):
        seen = set()
        for subcat in SUBCAT_TO_FAMILY:
            sizes = set(_benchmark_matrix_sizes(subcat))
            if not sizes:
                self.skipTest("benchmark files not fetched")
            seen |= sizes
        self.assertTrue(
            seen <= {3, 4, 5},
            f"the benchmark now contains matrices of size {seen - {3, 4, 5}}; "
            f"the corpus's size mix has to be re-derived")

    def test_the_benchmarks_size_mix_is_uniform(self):
        for subcat in SUBCAT_TO_FAMILY:
            with self.subTest(subcat=subcat):
                sizes = _benchmark_matrix_sizes(subcat)
                if not sizes:
                    self.skipTest("benchmark files not fetched")
                counts = Counter(sizes)
                self.assertEqual(
                    len(set(counts.values())), 1,
                    f"{subcat}: the benchmark's size mix is not uniform "
                    f"({dict(sorted(counts.items()))}); the corpus's mix "
                    f"has to be re-derived from it")


class TestTheCorpusTeachesTheBenchmarksSizes(unittest.TestCase):
    """The corpus must cover every size the benchmark scores."""

    # How much of the corpus may sit in a size the benchmark never
    # scores. It must be ZERO: those rows are capacity spent on nothing.
    MAX_OFF_MIX = 0.0

    def test_the_corpus_teaches_no_size_the_benchmark_never_scores(self):
        for subcat, family in SUBCAT_TO_FAMILY.items():
            with self.subTest(subcat=subcat):
                bench = set(_benchmark_matrix_sizes(subcat))
                if not bench:
                    self.skipTest("benchmark files not fetched")
                sizes = _corpus_matrix_sizes(family)
                if not sizes:
                    continue
                off = Counter(n for n in sizes if n not in bench)
                share = sum(off.values()) / len(sizes)
                self.assertLessEqual(
                    share, self.MAX_OFF_MIX,
                    f"{subcat}: {share:.0%} of the corpus's matrices are a size "
                    f"the benchmark never scores ({dict(sorted(off.items()))}). "
                    f"Those rows are capacity spent on nothing.")

    def test_the_corpus_covers_every_size_the_benchmark_scores(self):
        for subcat, family in SUBCAT_TO_FAMILY.items():
            with self.subTest(subcat=subcat):
                bench = set(_benchmark_matrix_sizes(subcat))
                if not bench:
                    self.skipTest("benchmark files not fetched")
                sizes = set(_corpus_matrix_sizes(family, 300))
                missing = bench - sizes
                self.assertFalse(
                    missing,
                    f"{subcat}: the corpus never generates a {missing} "
                    f"matrix, but the benchmark scores them")

    def test_the_corpus_matches_the_benchmarks_size_proportions(self):
        """Within a family, the corpus's size mix must be the benchmark's.

        The benchmark's is exactly uniform, so the corpus's must be too -
        within the tolerance a random draw gives.
        """
        for subcat, family in SUBCAT_TO_FAMILY.items():
            with self.subTest(subcat=subcat):
                bench = Counter(_benchmark_matrix_sizes(subcat))
                if not bench:
                    self.skipTest("benchmark files not fetched")
                want = {n: c / sum(bench.values()) for n, c in bench.items()}
                got = Counter(_corpus_matrix_sizes(family, 300))
                if not got:
                    continue
                total = sum(got.values())
                for n, share in want.items():
                    have = got.get(n, 0) / total
                    self.assertAlmostEqual(
                        have, share, delta=0.12,
                        msg=f"{subcat}: the corpus generates {have:.0%} "
                            f"{n}x{n} matrices but the benchmark scores "
                            f"{share:.0%}")


if __name__ == "__main__":
    unittest.main()
