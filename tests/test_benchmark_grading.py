"""Grade the real LinAlg-Bench answers, not a mock of them.

The benchmark is the number this project is judged on, and the code that
produces it had three separate bugs that made correct answers score zero:

  * the answer grammar had no rule for a bare comma-separated list, so all
    ninety eigenvalue references failed to parse;
  * the benchmark rounds its eigenvalues to four decimal places, and the
    comparison used a tolerance of 1e-9, so a correct eigenvalue was
    rejected;
  * \\boxed{...} was extracted with a non-greedy regex, which stopped at the
    first closing brace - the one inside \\begin{bmatrix} - so every
    matrix-valued answer was truncated to "\\begin{bmatrix". That was 420 of
    the 660 problems.

None of those would have been caught by a test with invented answers. This
test downloads the benchmark and checks the real thing. It skips if the
network is not available.
"""
import csv
import io
import os
import unittest
import urllib.request

from la.answer import answers_equal, parse_answer
from la.grading import extract_boxed, extract_model_answer, normalise_reference

SOURCES = {
    size: (f"https://raw.githubusercontent.com/linalgbench2026/LinAlgBench/"
           f"main/data/linalg_bench_{size}.csv")
    for size in ("3x3", "4x4", "5x5")
}
CACHE = os.path.join(os.path.dirname(__file__), "_cache")


def _fetch(size: str):
    path = os.path.join(CACHE, f"linalg_bench_{size}.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    try:
        req = urllib.request.Request(SOURCES[size], headers={"User-Agent": "vslm-test"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8")
    except Exception as exc:
        raise unittest.SkipTest(f"cannot download the benchmark: {exc}")
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def _all_rows():
    rows = []
    for size in SOURCES:
        rows.extend(csv.DictReader(io.StringIO(_fetch(size))))
    return rows


class TestBoxedExtraction(unittest.TestCase):
    def test_braces_are_balanced(self):
        raw = r"\boxed{A^T = \begin{bmatrix} 1 & 2 \end{bmatrix}}"
        self.assertEqual([r"A^T = \begin{bmatrix} 1 & 2 \end{bmatrix}"],
                         extract_boxed(raw))

    def test_escaped_braces_do_not_count(self):
        raw = r"\boxed{\{ 1, 2 \}}"
        self.assertEqual([r"\{ 1, 2 \}"], extract_boxed(raw))

    def test_the_last_box_wins(self):
        raw = r"first \boxed{1} then \boxed{2}"
        self.assertEqual(["1", "2"], extract_boxed(raw))


class TestBenchmarkGrading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = _all_rows()
        if not cls.rows:
            raise unittest.SkipTest("no benchmark rows")

    def test_every_reference_answer_parses(self):
        failures = []
        for row in self.rows:
            ref = row["answer_latex"]
            try:
                if parse_answer(normalise_reference(ref)) is None:
                    failures.append((row["Subcat"], ref))
            except Exception as exc:
                failures.append((row["Subcat"], f"{ref} -> {exc}"))
        self.assertEqual([], failures[:10],
                         f"{len(failures)} reference answers do not parse")

    def test_a_correct_answer_scores_correct(self):
        """Whatever shape a correct answer arrives in, it has to count."""
        failures = []
        for row in self.rows:
            ref = row["answer_latex"]
            try:
                truth = parse_answer(normalise_reference(ref))
            except Exception:
                continue
            for shape in (r"\boxed{" + ref + "}", ref, "### Answer\n" + ref):
                got = extract_model_answer(shape)
                try:
                    good = answers_equal(parse_answer(got), truth, tol=1e-3)
                except Exception:
                    good = False
                if not good:
                    failures.append((row["Subcat"], shape[:40], got[:40]))
        self.assertEqual([], failures[:10],
                         f"{len(failures)} correct answers scored wrong")

    def test_a_wrong_answer_scores_wrong(self):
        """The grader must not be so loose that anything passes."""
        accepted = []
        for row in self.rows:
            ref = row["answer_latex"]
            try:
                truth = parse_answer(normalise_reference(ref))
            except Exception:
                continue
            # perturb the reference by one
            wrong = str(ref).replace("=", "= 1 +", 1) if "=" in str(ref) else str(ref) + "1"
            try:
                got = parse_answer(normalise_reference(wrong))
                if answers_equal(got, truth, tol=1e-3):
                    accepted.append((row["Subcat"], ref))
            except Exception:
                pass
        self.assertEqual([], accepted[:10],
                         f"{len(accepted)} wrong answers were accepted")


if __name__ == "__main__":
    unittest.main()
