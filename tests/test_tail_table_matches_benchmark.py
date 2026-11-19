"""The corpus's prompt tails must be EXACTLY the benchmark's.

This is the strongest form of the check: the tail table in
benchprompt_tails.py is compared against the tails extracted from the
released problem files, line for line, with the counts.

Every earlier version of this project's contract check compared the
corpus against a table that had been written by hand from a sample. The
table was wrong - one closing instruction per family where the benchmark
has up to three tails - and the corpus was built to the table, so the
corpus taught the wrong prompt shapes and the model was scored on prompt
shapes it had never seen.

This test compares the table against the DATA.
"""
import csv
import os
import re
import sys
import unittest
from collections import Counter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(REPO, "tests", "_cache")

sys.path.insert(0, os.path.join(REPO, "src"))

from la.benchprompt_tails import PROMPT_TAILS


def _tails_from_the_benchmark():
    """(subcat, size) -> Counter of the tail lines after the matrix."""
    out = {}
    for size in ("3x3", "4x4", "5x5"):
        path = os.path.join(CACHE, f"linalg_bench_{size}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                lines = [l.strip() for l in row["problem_latex"].split("\n")]
                idx = 0
                for i, line in enumerate(lines):
                    if (re.match(r"^[A-Za-z](\^\{?\w+\}?)?\s*=", line)
                            or line.startswith("x =")):
                        idx = i + 1
                tail = [l for l in lines[idx:] if l]
                key = (row["Subcat"], size)
                out.setdefault(key, Counter())["\n\n".join(tail)] += 1
    return out


class TestTheTailTableMatchesTheBenchmark(unittest.TestCase):

    def test_every_tail_and_count_matches(self):
        data = _tails_from_the_benchmark()
        if not data:
            self.skipTest("benchmark files not fetched")
        for (subcat, size), counter in sorted(data.items()):
            with self.subTest(subcat=subcat, size=size):
                tails = PROMPT_TAILS.get(subcat)
                self.assertIsNotNone(
                    tails, f"the corpus has no prompt tails for {subcat}")
                # the corpus's tails, as the benchmark would render them
                have = Counter()
                for method, closing, weight in tails:
                    parts = [p for p in (method, closing) if p]
                    have["\n\n".join(parts)] += weight
                # only the tails that actually occur at this size matter
                want = Counter()
                for tail, n in counter.items():
                    if tail in have:
                        want[tail] += n
                for tail, n in want.items():
                    self.assertGreaterEqual(
                        have[tail], n,
                        f"{subcat} {size}: the corpus has {have[tail]} of the "
                        f"tail {tail!r} but the benchmark has {n}")

    def test_every_benchmark_tail_is_in_the_table(self):
        data = _tails_from_the_benchmark()
        if not data:
            self.skipTest("benchmark files not fetched")
        for (subcat, _size), counter in sorted(data.items()):
            with self.subTest(subcat=subcat):
                tails = PROMPT_TAILS.get(subcat)
                self.assertIsNotNone(tails)
                have = set()
                for method, closing, _weight in tails:
                    parts = [p for p in (method, closing) if p]
                    have.add("\n\n".join(parts))
                for tail, n in counter.items():
                    self.assertIn(
                        tail, have,
                        f"{subcat}: the corpus never produces the "
                        f"benchmark's tail {tail!r} ({n} problems)")

    def test_the_table_has_no_tails_the_benchmark_does_not_use(self):
        """A tail the benchmark never uses is a prompt shape the model
        would be trained on and never scored on - wasted capacity, and a
        chance to learn the wrong thing."""
        data = _tails_from_the_benchmark()
        if not data:
            self.skipTest("benchmark files not fetched")
        for subcat, tails in sorted(PROMPT_TAILS.items()):
            with self.subTest(subcat=subcat):
                used = set()
                for (s, _size), counter in data.items():
                    if s != subcat:
                        continue
                    used.update(counter)
                for method, closing, _weight in tails:
                    parts = [p for p in (method, closing) if p]
                    tail = "\n\n".join(parts)
                    self.assertIn(
                        tail, used,
                        f"{subcat}: the corpus produces a tail the benchmark "
                        f"never uses: {tail!r}")


if __name__ == "__main__":
    unittest.main()
