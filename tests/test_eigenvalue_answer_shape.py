"""The eigenvalue answer must not be teachable as real/imaginary PAIRS.

The corpus's plot spec for the spectrum is
`{"values": [[re, im], [re, im], ...]}` — a list of pairs. The ANSWER for the
same problem is a bare list of numbers. The two shapes sit three lines apart
in every training example, and a 1.7B model learns the wrong one: it writes
`3.9, 0.0, 1.2, 0.0` where the grader expects `3.9, 1.2`.

The grader reads a 6-element list against a 3-element reference and calls it
wrong. Measured against the real grader, that comparison is False, so every
such answer scores zero even when the mathematics is right.

The fix: the spectrum plot for a real spectrum must use the SAME plain
number list the answer uses. The renderer already accepts a plain number for
a real eigenvalue, and the schema's `complex_list` shape accepts it too.
"""
import random
import unittest

from la.answer import answers_equal, parse_answer
from la.problems import generate
from la.schema import validate_spec


class TestEigenvalueAnswerIsNotTaughtAsPairs(unittest.TestCase):

    def test_the_exact_family_plots_plain_numbers(self):
        p = generate("eigenvalue", random.Random(21))
        self.assertIsNotNone(p)
        values = p.spec["data"]["values"]
        self.assertTrue(all(isinstance(v, (int, float)) for v in values),
                        f"the spectrum must plot plain numbers, got {values}")

    def test_the_numeric_family_plots_plain_numbers(self):
        p = generate("eigenvalue_numeric", random.Random(22))
        self.assertIsNotNone(p)
        values = p.spec["data"]["values"]
        self.assertTrue(all(isinstance(v, (int, float)) for v in values),
                        f"the spectrum must plot plain numbers, got {values}")

    def test_the_spec_still_validates(self):
        for fam in ("eigenvalue", "eigenvalue_numeric"):
            p = generate(fam, random.Random(23))
            self.assertEqual(validate_spec(p.spec), [], fam)

    def test_the_answer_shape_and_the_plot_shape_agree(self):
        """The two lists in one example must have the same length.

        The answer is what the grader reads; the plot is what the model
        copies. If they are different lengths, the model has been shown two
        shapes for the same thing, and it picks the wrong one.
        """
        for fam in ("eigenvalue", "eigenvalue_numeric"):
            with self.subTest(family=fam):
                p = generate(fam, random.Random(24))
                answer_values = parse_answer(p.answer)
                plot_values = p.spec["data"]["values"]
                self.assertEqual(len(answer_values), len(plot_values),
                                 f"{fam}: the answer has "
                                 f"{len(answer_values)} values and the plot "
                                 f"has {len(plot_values)}")

    def test_a_pairs_answer_does_not_score(self):
        """The failure this prevents, as an assertion."""
        pairs = parse_answer("3.9576, 0.0, 0.1215, 0.0, -2.0791, 0.0")
        ref = parse_answer("3.9576, 0.1215, -2.0791")
        self.assertFalse(answers_equal(pairs, ref),
                         "a pairs answer must not score against a bare list")


if __name__ == "__main__":
    unittest.main()
