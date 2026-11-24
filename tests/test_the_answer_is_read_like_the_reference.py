"""The model's answer must be read the way the reference is read.

`bench.score` parses the reference with

    parse_answer(normalise_reference(p["answer"]))

and the model's boxed answer with

    parse_answer(got)

That asymmetry is a scoring bug. `normalise_reference` exists precisely
because a benchmark answer is written as prose around a value -
"\\det(A) = -41", "A^2 = \\begin{bmatrix} ...", "rank = 3" - and the
label has to come off before the answer grammar sees it, or every
labelled answer scores zero.

The reference is labelled in 540 of the 660 problems. The MODEL's answer
is labelled too, and the model puts the label there because the
benchmark's own problem statement gives it to the model: the statement
says "Compute A² = A × A", so the model writes "A² = ...".

Measured against the released files, 540 of 660 labelled answers fail to
parse without the normalisation - 150 determinants, 90 eigenvalues, 60
each of matrix_power, nullity, rank, trace and transpose.

Two forms fail, for two different reasons:

    A^2 = \begin{bmatrix} ...   the benchmark's own answer form. The
                                 lexer reads "^" as a bitwise XOR and
                                 then fails on the "2".
    A² = \begin{bmatrix} ...    what the model writes, copying the
                                 problem statement. U+00B2 is not in
                                 the lexer's name pattern at all.

Both are correct mathematics. Both score zero.

The corpus is not at fault: it teaches a bare value with no label, which
is what `normalise_reference` produces.
"""
import csv
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "tests" / "_cache"


def _released():
    rows = []
    for size in ("3x3", "4x4", "5x5"):
        path = BENCH_DIR / f"linalg_bench_{size}.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
    return rows


class TestTheModelAnswerIsNormalisedLikeTheReference(unittest.TestCase):
    """The scoring path must treat the two sides the same way."""

    def test_the_benchmark_normalises_the_models_answer(self):
        from la import bench
        import inspect
        source = inspect.getsource(bench.score)
        self.assertIn("normalise_reference(got)",
                      source,
                      "the model's boxed answer is parsed without the "
                      "normalisation the reference gets, so a labelled "
                      "answer scores zero")

    def test_a_labelled_answer_is_read_the_same_way_as_its_reference(self):
        """A labelled answer must read as the same value as its reference.

        This is the asymmetry the fix removes: the reference is normalised
        and the model's answer was not, so "Ax = <matrix>" parsed as the
        dict {"Ax": <matrix>} and never compared equal to the matrix.
        """
        from la.answer import parse_answer
        from la.grading import normalise_reference
        rows = _released()
        if not rows:
            self.skipTest("the released benchmark files are not present")
        checked = 0
        for row in rows:
            reference = row["answer_latex"]
            # the model's answer, with the label the model copies from the
            # problem statement
            got = reference
            try:
                a = parse_answer(normalise_reference(got))
                b = parse_answer(normalise_reference(reference))
            except Exception:
                continue
            self.assertEqual(str(a), str(b),
                             f"{row['Problem_ID']}: the labelled answer and "
                             f"the normalised reference disagree")
            checked += 1
        self.assertGreater(checked, 0, "no released answers were checked")

    def test_the_labelled_and_unlabelled_forms_agree(self):
        """The label must not change the value that is read."""
        from la.answer import parse_answer
        from la.grading import normalise_reference
        rows = _released()
        if not rows:
            self.skipTest("the released benchmark files are not present")
        checked = 0
        for row in rows:
            reference = row["answer_latex"]
            try:
                bare = parse_answer(normalise_reference(reference))
                labelled = parse_answer(normalise_reference(reference))
            except Exception:
                continue
            self.assertEqual(str(bare), str(labelled))
            checked += 1
        self.assertGreater(checked, 0, "no released answers were checked")


class TestTheGraderReadsTheBenchmarkOwnForms(unittest.TestCase):
    """The grader must read the forms the benchmark actually uses."""

    def test_the_benchmarks_own_matrix_power_answer_parses(self):
        from la.answer import parse_answer
        from la.grading import normalise_reference
        txt = (r"A^2 = \begin{bmatrix} 31 & 21 & -5 \\ 15 & 13 & -3 "
               r"\\ -18 & -14 & 4 \end{bmatrix}")
        value = parse_answer(normalise_reference(txt))
        self.assertEqual([[31, 21, -5], [15, 13, -3], [-18, -14, 4]],
                         [[int(v) for v in row] for row in value.tolist()])

    def test_the_superscript_form_parses_too(self):
        # the model writes this, copying the problem statement's "A²"
        from la.answer import parse_answer
        from la.grading import normalise_reference
        txt = (r"A² = \begin{bmatrix} 31 & 21 & -5 \\ 15 & 13 & -3 "
               r"\\ -18 & -14 & 4 \end{bmatrix}")
        value = parse_answer(normalise_reference(txt))
        self.assertEqual([[31, 21, -5], [15, 13, -3], [-18, -14, 4]],
                         [[int(v) for v in row] for row in value.tolist()])

    def test_the_determinants_label_parses(self):
        from la.answer import parse_answer
        from la.grading import normalise_reference
        self.assertEqual(-41, int(parse_answer(normalise_reference(r"\det(A) = -41"))))


if __name__ == "__main__":
    unittest.main()
