"""The progress line.

A long generation that prints nothing looks exactly like a hang, and the
first version of this printed every five batches, which on the benchmark
was twelve minutes of silence between lines.
"""

import re
import unittest

from la.progress import progress_line


def _count(line: str) -> int:
    m = re.search(r"(\d+)/(\d+)", line)
    assert m, f"no count in {line!r}"
    return int(m.group(1))


class TestProgressLine(unittest.TestCase):
    def test_it_reports_the_count_the_rate_and_the_time_left(self):
        line = progress_line("benchmark", 40, 660, 720.0)
        self.assertIn("40/660", line)
        self.assertIn("benchmark", line)
        self.assertIn("min left", line)
        self.assertIn("min elapsed", line)

    def test_the_rate_is_problems_per_second(self):
        line = progress_line("x", 10, 100, 10.0)
        self.assertIn("1.0/s", line)

    def test_it_does_not_divide_by_zero_before_the_first_batch(self):
        line = progress_line("x", 0, 660, 0.0)
        self.assertIn("0/660", line)

    def test_it_does_not_claim_a_time_left_before_it_has_measured_one(self):
        line = progress_line("x", 0, 660, 60.0)
        self.assertIn("about ? min left", line)

    def test_a_meaningless_rate_does_not_produce_a_meaningless_estimate(self):
        """One problem in a second is a rate of one, but one problem in an
        hour is not worth extrapolating from."""
        line = progress_line("x", 1, 660, 3600.0)
        self.assertIn("about ? min left", line)
        line = progress_line("x", 1, 660, 1.0)
        self.assertNotIn("about ? min left", line)

    def test_the_batch_in_flight_is_reported(self):
        """The problem count only moves when a whole batch finishes, so
        without the batch number the line looks frozen."""
        line = progress_line("x", 0, 660, 60.0, batch=4, batches=83)
        self.assertIn("batch 4/83", line)

    def test_the_count_only_ever_grows(self):
        seen = [progress_line("x", d, 660, 60.0 * (d + 1))
                for d in range(0, 660, 40)]
        counts = [_count(s) for s in seen]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[-1], 640)
