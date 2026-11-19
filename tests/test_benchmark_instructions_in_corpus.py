"""The corpus must contain the benchmark's OWN method instructions.

This is the defect that cost run 6 its determinant column.

LinAlg-Bench's problems are not all "find X of the matrix". Some of them
name the METHOD:

    determinant         18 of 150  "Use cofactor expansion along the first row."
    eigenvalue          24 of  90  "compute the characteristic polynomial ..."
    multiplication      12 of  60  "use row reduction ... show your row operations"
    nullity             16 of  60  "use row reduction ... show your row operations"
    rank                14 of  60  "use row reduction ... show your row operations"

That instruction is part of the PROMPT. The model is scored on a prompt
that says "use row reduction and show your row operations" - and the
corpus never produced a prompt that said anything of the kind. The corpus
taught the method; it never taught the model that the method would be
requested in the prompt.

That is why run 6 scored 0/150 on determinant while the BASE model, which
had never seen the method instruction either, scored 54/150: the base
model falls back on what it knows, and the fine-tune had been taught a
narrow, deterministic derivation it could only reproduce.

So the corpus has to contain prompts carrying the benchmark's own
instruction strings, with a derivation that answers them.
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
                rows.append(row)
    return rows


def _instructions_for(subcat):
    """The distinct method instructions the benchmark uses for a subcat.

    These are the tail lines that are NOT the closing instruction - the
    closing line is the one that mentions \\boxed{} - and NOT a matrix
    definition.
    """
    out = set()
    for r in _benchmark_rows():
        if r["Subcat"] != subcat:
            continue
        for line in r["problem_latex"].split("\n"):
            line = line.strip()
            if not line or "boxed" in line:
                continue
            if "\\begin{" in line:
                # a matrix definition
                continue
            out.add(line)
    return out


class TestTheBenchmarkNamesAMethod(unittest.TestCase):

    def test_the_benchmark_really_does_name_methods(self):
        """If this stops being true, the fix below stops being needed."""
        rows = _benchmark_rows()
        if not rows:
            self.skipTest("benchmark files not fetched")
        det = _instructions_for("determinant")
        self.assertTrue(
            any("cofactor expansion" in d for d in det),
            "the benchmark no longer instructs cofactor expansion")
        rank = _instructions_for("rank")
        self.assertTrue(
            any("row reduction" in d for d in rank),
            "the benchmark no longer instructs row reduction")


class TestTheCorpusProducesTheBenchmarksInstructions(unittest.TestCase):

    def _sample(self, family, n=400):
        rng = random.Random(97)
        out = []
        for _ in range(n):
            p = generate(family, rng)
            if p is not None:
                out.append(p)
        return out

    def test_the_corpus_produces_the_cofactor_instruction(self):
        rows = _benchmark_rows()
        if not rows:
            self.skipTest("benchmark files not fetched")
        wanted = [d for d in _instructions_for("determinant")
                  if "cofactor expansion" in d]
        self.assertTrue(wanted)
        problems = self._sample("determinant")
        seen = 0
        for p in problems:
            if any(w in p.question for w in wanted):
                seen += 1
        self.assertGreater(seen, 0,
                           "the corpus never produces the benchmark's "
                           "cofactor instruction")

    def test_the_corpus_produces_the_row_reduction_instruction(self):
        rows = _benchmark_rows()
        if not rows:
            self.skipTest("benchmark files not fetched")
        for family in ("rank", "nullity"):
            wanted = [d for d in _instructions_for(family)
                      if "row reduction" in d]
            self.assertTrue(wanted, f"{family} has no instruction")
            problems = self._sample(family)
            seen = sum(1 for p in problems
                       if any(w in p.question for w in wanted))
            self.assertGreater(seen, 0,
                               f"the corpus never produces the benchmark's "
                               f"{family} instruction")

    def test_the_corpus_produces_the_characteristic_polynomial_instruction(self):
        rows = _benchmark_rows()
        if not rows:
            self.skipTest("benchmark files not fetched")
        wanted = [d for d in _instructions_for("eigenvalue")
                  if "characteristic polynomial" in d]
        self.assertTrue(wanted)
        problems = self._sample("eigenvalue")
        seen = sum(1 for p in problems
                   if any(w in p.question for w in wanted))
        self.assertGreater(seen, 0,
                           "the corpus never produces the benchmark's "
                           "characteristic-polynomial instruction")

    def test_the_corpus_produces_the_multiplication_instruction(self):
        rows = _benchmark_rows()
        if not rows:
            self.skipTest("benchmark files not fetched")
        wanted = [d for d in _instructions_for("multiplication")
                  if "row-by-column" in d]
        self.assertTrue(wanted)
        problems = self._sample("matrix_multiply")
        seen = sum(1 for p in problems
                   if any(w in p.question for w in wanted))
        self.assertGreater(seen, 0,
                           "the corpus never produces the benchmark's "
                           "multiplication instruction")


if __name__ == "__main__":
    unittest.main()
