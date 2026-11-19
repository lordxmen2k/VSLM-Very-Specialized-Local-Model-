"""The corpus must teach the prompt shape the 4x4 file arrives in.

All 220 of LinAlg-Bench's 4x4 problems are a bare matrix with no task
statement and no closing instruction. The harness reconstructs the task
from the subcategory label, so the model is scored on:

    Solve this linear algebra problem. Show your work and give the final
    answer.

    Find the determinant of the 4x4 matrix A.

    A = \\begin{bmatrix} ... \\end{bmatrix}

with no closing instruction at all. The corpus's question always has one,
so a third of the benchmark is scored on a prompt shape the model was never
trained on.

Run 6's 4x4 column was 37.7% against the base model's 41.8% - a loss on
the size where the whole column is out of distribution.

The corpus has to contain examples of the bare-matrix shape, or the model
has to be trained to answer without being told what to put in the box.
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


def _rows():
    out = []
    for size in ("3x3", "4x4", "5x5"):
        path = os.path.join(CACHE, f"linalg_bench_{size}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out.append((size, row))
    return out


class TestThe4x4FileHasNoClosingInstruction(unittest.TestCase):

    def test_every_4x4_problem_is_a_bare_matrix(self):
        rows = _rows()
        if not rows:
            self.skipTest("benchmark files not fetched")
        four = [(s, r) for s, r in rows if s == "4x4"]
        self.assertEqual(len(four), 220)
        for _s, r in four:
            self.assertNotIn("Show all intermediate steps",
                             r["problem_latex"])
            self.assertNotIn("boxed", r["problem_latex"])

    def test_the_harness_reconstructs_the_task_for_them(self):
        from la.benchprompt import benchmark_prompt_pair
        rows = _rows()
        if not rows:
            self.skipTest("benchmark files not fetched")
        r = [x for s, x in rows if s == "4x4"][0]
        system, user = benchmark_prompt_pair(r["problem_latex"],
                                             r["Subcat"], "4x4")
        self.assertIn("Solve this linear algebra problem", user)
        self.assertNotIn("Show all intermediate steps", user)


class TestTheCorpusHasExamplesWithoutAClosingInstruction(unittest.TestCase):

    def test_some_corpus_examples_have_no_closing_instruction(self):
        """A share of the corpus must be the bare-matrix shape."""
        rng = random.Random(41)
        without = 0
        total = 0
        for _ in range(300):
            p = generate("determinant", rng)
            if p is None:
                continue
            total += 1
            if "Show all intermediate steps" not in p.question:
                without += 1
        self.assertGreater(total, 0)
        self.assertGreater(without / total, 0.15,
                           "too few corpus examples carry the bare-matrix "
                           "shape the 4x4 file arrives in")


if __name__ == "__main__":
    unittest.main()
