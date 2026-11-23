"""The rank and nullity families must cover the benchmark's rank range.

THE DEFECT THIS FILE GUARDS AGAINST. The corpus's rank and nullity
matrices were built with

    r = rng.randint(1, n - 1)

so the rank was drawn from 1 to n-1 and the corpus COULD NOT PRODUCE A
FULL-RANK MATRIX. Not rarely - never.

The benchmark's rank and nullity matrices span the full range, and the
full-rank case is the common one:

    benchmark rank  (size, rank)     corpus rank  (size, rank)
      3x3  1: 2  2: 8  3: 10          3x3  1: 51  2: 43
      4x4  2: 4  3: 7  4: 9           4x4  1: 36  2: 32  3: 34
      5x5  2: 3  3: 4  4: 6  5: 7     5x5  1: 26  2: 26  3: 24  4: 28

    benchmark nullity (size, rank)    corpus nullity (size, rank)
      3x3  1: 12 2: 7  3: 1           3x3  1: 57  2: 54
      4x4  1: 3  2: 5  3: 7  4: 5     4x4  1: 29  2: 34  3: 27
      5x5  2: 2  3: 5  4: 7  5: 6     5x5  1: 21  2: 23  3: 27  4: 28

The corpus's distribution is the benchmark's shifted one rank down. The
benchmark's single most common 3x3 rank answer is 3, and the corpus
cannot produce a matrix of rank 3 at 3x3.

The rank family scored 27 of 60 at run 10 against the base model's 21,
and the nullity family 14 of 60 against the base model's 19 - the two
families that LOST to the base model, and they are the two families
whose corpus was missing a whole rank.

THE RANK MIX IS THE BENCHMARK'S MIX. The test compares the corpus's
rank distribution against the benchmark's, per size, and requires the
corpus to reach the benchmark's maximum rank at every size.
"""
from __future__ import annotations

import collections
import csv
import os
import random
import re
import unittest

import sympy as sp

from la.problems import generate

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(REPO, "tests", "_cache")


def _benchmark_ranks(subcat):
    """(size, rank) -> count, from the released files."""
    out = collections.Counter()
    for size in ("3x3", "4x4", "5x5"):
        path = os.path.join(CACHE, f"linalg_bench_{size}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["Subcat"] != subcat:
                    continue
                for m in re.finditer(
                        r"\\begin\{[bp]matrix\}(.*?)\\end\{[bp]matrix\}",
                        row["problem_latex"], re.S):
                    rows = [ln for ln in m.group(1).split("\\\\") if ln.strip()]
                    mat = [[int(x) for x in re.findall(r"-?\d+", r)]
                           for r in rows]
                    if mat and all(len(r) == int(size[0]) for r in mat):
                        out[(size, sp.Matrix(mat).rank())] += 1
    return out


def _corpus_ranks(family, n=300, seed=4242):
    """(size, rank) -> count, from the corpus."""
    out = collections.Counter()
    rng = random.Random(seed)
    for _ in range(n):
        p = generate(family, rng)
        if p is None:
            continue
        for m in re.finditer(r"\\begin\{[bp]matrix\}(.*?)\\end\{[bp]matrix\}",
                              p.question, re.S):
            rows = [ln for ln in m.group(1).split("\\\\") if ln.strip()]
            mat = [[int(x) for x in re.findall(r"-?\d+", r)] for r in rows]
            if mat and all(len(r) == len(mat[0]) for r in mat):
                out[(f"{len(mat)}x{len(mat)}", sp.Matrix(mat).rank())] += 1
    return out


class TestTheCorpusCoversTheBenchmarksRanks(unittest.TestCase):

    def test_the_corpus_reaches_the_benchmarks_maximum_rank(self):
        """A full-rank matrix must be reachable at every size.

        The corpus drew its rank from 1 to n-1, so a full-rank matrix
        was impossible. The benchmark's most common 3x3 rank answer is 3.
        """
        for family, subcat in (("rank", "rank"), ("nullity", "nullity")):
            bench = _benchmark_ranks(subcat)
            if not bench:
                self.skipTest("benchmark files not fetched")
            corpus = _corpus_ranks(family)
            for size in ("3x3", "4x4", "5x5"):
                with self.subTest(family=family, size=size):
                    want_max = max(r for (s, r) in bench if s == size)
                    have = {r for (s, r) in corpus if s == size}
                    self.assertIn(
                        want_max, have,
                        f"{family} at {size}: the corpus reaches ranks "
                        f"{sorted(have)} but the benchmark's maximum rank "
                        f"is {want_max}, and it is the common case there")


if __name__ == "__main__":
    unittest.main()
