"""The task statement must be the BENCHMARK'S, verbatim, for each size.

RUN 10 SCORED 276/660 = 41.8%. The run-10 fix - restoring the products to
the 4x4 dot products - worked exactly as intended:

    matrix_power    15 -> 38   (base 34)
    matrix_vector    8 -> 43   (base 44)
    multiplication  29 -> 35   (base 25)

and the overall score went 225 -> 276.

Two families are still broken, and they are the two the user singled out:

    family        run 10   base
    determinant       1/150   54/150
    nullity          14/60    19/60

THE CAUSE: the task statement.

Measuring the released benchmark files shows that LinAlg-Bench's task
statement is DETERMINISTIC per (family, size):

    3x3   one specific phrasing, naming the size
    4x4   a BARE MATRIX - no task statement at all
    5x5   one specific phrasing, naming the size

    determinant   3x3  "Find the determinant of the 3×3 matrix A."
                  4x4  (bare matrix, no task)
                  5x5  "Find the determinant of the 5×5 matrix A."
    rank          3x3  "Find the rank of the 3×3 matrix A."
                  4x4  (bare matrix, no task)
                  5x5  "Find the rank of the 5×5 matrix A."
    nullity       ... the same shape
    trace         3x3  "Compute the trace of the 3×3 matrix A."
    transpose     3x3  "Find the transpose of the 3×3 matrix A."
    matrix_power  3x3  "Compute A² = A × A for the 3×3 matrix A."
    matrix_vector 3x3  "Compute the matrix-vector product b = A·x for ..."
    multiplication 3x3 "Compute the matrix product C = A × B for ..."

Two things are wrong in the corpus's task statements, and both are
measurable:

1. **The corpus invents its own wordings.** For the determinant it uses
   six different phrasings, and 66-78 per cent of them name NO SIZE:

       determinant  233/300 (78%) "task, no size named"
                   30/300 (10%) names 3x3
                   19/300  (6%) names 5x5
                   18/300  (6%) names 4x4

   The benchmark names the size in EVERY 3x3 and 5x5 question.

2. **The corpus writes the size with an ASCII "x" and the benchmark
   writes it with U+00D7.** These are different strings:

       corpus:     "Find the determinant of the 3x3 matrix A."
       benchmark:  "Find the determinant of the 3×3 matrix A."

   The model is trained on one and scored on the other.

So this module pins the task statement against the RELEASED benchmark
files, for every (family, size), in both directions:

  - the corpus must produce the benchmark's exact task statement;
  - the corpus must not produce a task statement the benchmark never uses.
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

# The corpus's family -> the benchmark's subcategory.
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

# The benchmark's own families, for the reverse direction.
FAMILY_TO_SUBCAT = {
    "determinant": "determinant",
    "eigenvalue": "eigenvalue",
    "eigenvalue_numeric": "eigenvalue",
    "matrix_multiply": "multiplication",
    "matrix_power": "matrix_power",
    "matrix_trace": "trace",
    "matrix_transpose": "transpose",
    "matrix_vector": "matrix_vector",
    "nullity": "nullity",
    "rank": "rank",
}


def _benchmark_task_statements(subcat):
    """The task statement the benchmark uses, per size.

    Returns {size: Counter(task_statement)}. The bare-matrix shape is
    recorded as "" - the benchmark's 4x4 questions carry no task at all.
    """
    out = {}
    for size in ("3x3", "4x4", "5x5"):
        path = os.path.join(CACHE, f"linalg_bench_{size}.csv")
        if not os.path.exists(path):
            continue
        counts = Counter()
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["Subcat"] != subcat:
                    continue
                counts[_first_paragraph(row["problem_latex"])] += 1
        out[size] = counts
    return out


def _first_paragraph(prompt):
    """The benchmark's task paragraph, or "" for the bare-matrix shape."""
    first = prompt.split("\n\n")[0].strip()
    if first.startswith("A =") or first.startswith("A="):
        return ""
    return first


def _corpus_task_statements(family, n=300):
    """The task paragraph the corpus produces, per size."""
    rng = random.Random(4242)
    out = {"3x3": Counter(), "4x4": Counter(), "5x5": Counter()}
    for _ in range(n):
        p = generate(family, rng)
        if p is None:
            continue
        parts = [x.strip() for x in p.question.split("\n\n")]
        # parts[0] is the preamble, parts[1] is the task or the matrix
        body = parts[1] if len(parts) > 1 else ""
        task = body.split("\n\n")[0].strip() if body else ""
        if task.startswith("A =") or task.startswith("A="):
            task = ""
        # which size is this problem? the matrix's own dimension
        m = re.search(r"\\begin\{[bp]matrix\}(.*?)\\end\{[bp]matrix\}",
                      p.question, re.S)
        if not m:
            continue
        rows = [ln for ln in m.group(1).split("\\\\") if ln.strip()]
        if not rows:
            continue
        size = f"{len(rows)}x{len(rows)}"
        if size in out:
            out[size][task] += 1
    return out


class TestTheBenchmarkTaskStatementsAreWhatTheyAre(unittest.TestCase):
    """The premise. If the benchmark changes, everything below moves."""

    def test_the_benchmark_names_the_size_in_its_3x3_and_5x5_questions(self):
        for subcat in SUBCAT_TO_FAMILY:
            with self.subTest(subcat=subcat):
                per_size = _benchmark_task_statements(subcat)
                if not per_size:
                    self.skipTest("benchmark files not fetched")
                for size in ("3x3", "5x5"):
                    counts = per_size.get(size) or Counter()
                    if not counts:
                        continue
                    for task, n in counts.items():
                        self.assertTrue(
                            task,
                            f"{subcat} {size}: the benchmark has a "
                            f"question with no task statement")
                        self.assertIn(
                            size[0] + "×" + size[2],
                            task,
                            f"{subcat} {size}: the benchmark's task "
                            f"{task!r} does not name the size")

    def test_the_benchmark_uses_the_multiplication_sign(self):
        """U+00D7, not an ASCII "x". The corpus used "3x3"."""
        for subcat in ("determinant", "rank", "nullity"):
            per_size = _benchmark_task_statements(subcat)
            for size, counts in per_size.items():
                for task in counts:
                    if task and re.search(r"\d\s*x\s*\d", task):
                        self.fail(f"{subcat} {size}: the benchmark writes "
                                  f"{task!r} with an ASCII x")


class TestTheCorpusUsesTheBenchmarksTaskStatements(unittest.TestCase):
    """The corpus's task statement must be the benchmark's."""

    def test_the_corpus_produces_the_benchmarks_task_statement(self):
        """For the 3x3 and 5x5 sizes the corpus must produce the
        benchmark's own statement, verbatim.

        The 4x4 size is different on purpose. The benchmark's 4x4 FILE is
        a bare matrix, but the SCORER reconstructs the task from the
        subcategory label and sends that - so what the corpus has to
        teach is the reconstructed statement, not an empty one. The
        scorer's prompt is the contract, and it is checked separately by
        TestTheCorpusPromptMatchesTheScorersPrompt.
        """
        for subcat, family in SUBCAT_TO_FAMILY.items():
            with self.subTest(subcat=subcat):
                bench = _benchmark_task_statements(subcat)
                corpus = _corpus_task_statements(family)
                if not bench:
                    self.skipTest("benchmark files not fetched")
                for size, counts in bench.items():
                    if not counts or size == "4x4":
                        continue
                    want = set(counts)
                    got = set(corpus.get(size) or Counter())
                    missing = want - got
                    self.assertFalse(
                        missing,
                        f"{subcat} {size}: the corpus never produces the "
                        f"benchmark's task statement {sorted(missing)!r}")

    def test_the_corpus_produces_no_task_statement_the_benchmark_never_uses(self):
        """Again, for the 3x3 and 5x5 sizes.

        The 4x4 statement is the scorer's reconstructed one, so the
        benchmark's file - which has no statement at all - is not the
        right yardstick for it.
        """
        for subcat, family in SUBCAT_TO_FAMILY.items():
            with self.subTest(subcat=subcat):
                bench = _benchmark_task_statements(subcat)
                corpus = _corpus_task_statements(family)
                if not bench:
                    self.skipTest("benchmark files not fetched")
                for size, counts in bench.items():
                    if not counts or size == "4x4":
                        continue
                    want = set(counts)
                    got = set(corpus.get(size) or Counter())
                    extra = got - want
                    self.assertFalse(
                        extra,
                        f"{subcat} {size}: the corpus produces task "
                        f"statements the benchmark never uses: "
                        f"{sorted(extra)!r}")

    def test_the_corpus_names_the_size_the_way_the_benchmark_does(self):
        """The corpus wrote '3x3' where the benchmark writes '3×3'."""
        for subcat, family in SUBCAT_TO_FAMILY.items():
            with self.subTest(subcat=subcat):
                corpus = _corpus_task_statements(family)
                for size, counts in corpus.items():
                    for task in counts:
                        if re.search(r"\d\s*x\s*\d", task):
                            self.fail(
                                f"{subcat} {size}: the corpus writes "
                                f"{task!r} with an ASCII 'x', but the "
                                f"benchmark writes the size with "
                                f"U+00D7")


class TestTheCorpusPromptMatchesTheScorersPrompt(unittest.TestCase):
    """The corpus's prompt and the SCORER's prompt must be the same shape.

    The corpus's question is its own prompt - `model_prompt()` returns it
    unchanged - and the scorer builds the benchmark's prompt with
    `benchmark_prompt_pair()`. If the two differ in structure, the model
    is trained on one shape and scored on another.

    The shapes are:

        3x3   preamble, task, matrix, closing instruction   (4 paragraphs)
        4x4   preamble, task, matrix                        (3 paragraphs)
        5x5   preamble, task, matrix, closing instruction   (4 paragraphs)

    The benchmark's own 4x4 file is a BARE MATRIX - one paragraph - but
    the scorer reconstructs the task from the subcategory label and sends
    that, so the corpus has to teach the reconstructed statement.
    """

    def test_the_corpus_and_the_scorer_agree_on_the_prompt_shape(self):
        from la.benchprompt import benchmark_prompt_pair
        for subcat, family in SUBCAT_TO_FAMILY.items():
            for size in ("3x3", "4x4", "5x5"):
                with self.subTest(subcat=subcat, size=size):
                    bench = _benchmark_task_statements(subcat)
                    if not bench or size not in bench:
                        self.skipTest("benchmark files not fetched")
                    _system, user = benchmark_prompt_pair(
                        _benchmark_prompt(subcat, size), subcat, size)
                    want = [x for x in user.split("\n\n") if x]

                    corpus = _corpus_task_statements(family, n=400)
                    got = _corpus_prompt_for(family, size)
                    if got is None:
                        continue
                    got_paras = [x for x in got.split("\n\n") if x]
                    self.assertEqual(
                        len(want), len(got_paras),
                        f"{subcat} {size}: the scorer sends {len(want)} "
                        f"paragraphs and the corpus sends {len(got_paras)}")

    def test_the_corpus_and_the_scorer_agree_on_the_non_matrix_paragraphs(self):
        """The SETS of paragraphs must agree.

        Not one sample to one sample: the trace's 3x3 question comes in
        two forms - "Find the trace of the 3×3 matrix A." (6 of 20) and
        "Compute the trace of the 3×3 matrix A." (14 of 20) - and so
        does the corpus, at the same proportion. A test that compared one
        corpus sample to one benchmark sample would fail on the form it
        happened not to pick.
        """
        from la.benchprompt import benchmark_prompt_pair
        for subcat, family in SUBCAT_TO_FAMILY.items():
            for size in ("3x3", "4x4", "5x5"):
                with self.subTest(subcat=subcat, size=size):
                    bench = _benchmark_task_statements(subcat)
                    if not bench or size not in bench:
                        self.skipTest("benchmark files not fetched")
                    want = set()
                    for prompt in _benchmark_prompts(subcat, size):
                        _system, user = benchmark_prompt_pair(
                            prompt, subcat, size)
                        for par in user.split("\n\n"):
                            if par and not _is_matrix_paragraph(par):
                                want.add(par)
                    got = set()
                    for prompt in _corpus_prompts(family, size):
                        for par in prompt.split("\n\n"):
                            if par and not _is_matrix_paragraph(par):
                                got.add(par)
                    if not got:
                        continue
                    self.assertEqual(
                        want, got,
                        f"{subcat} {size}: the scorer and the corpus do not "
                        f"send the same set of paragraphs")


def _is_matrix_paragraph(par):
    """Is this paragraph a matrix or a vector?

    The multiplication family's body is two matrices in one paragraph, so
    filtering on "A =" alone would leave the "B = ..." half in.
    """
    return "\\begin{" in par


def _benchmark_prompts(subcat, size, limit=60):
    """The benchmark's own prompts for a subcategory and size."""
    path = os.path.join(CACHE, f"linalg_bench_{size}.csv")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["Subcat"] == subcat:
                out.append(row["problem_latex"])
                if len(out) >= limit:
                    break
    return out


def _corpus_prompts(family, size, limit=60):
    """The corpus's prompts at a given size."""
    rng = random.Random(4242)
    out = []
    for _ in range(600):
        p = generate(family, rng)
        if p is None:
            continue
        m = re.search(r"\\begin\{[bp]matrix\}(.*?)\\end\{[bp]matrix\}",
                      p.question, re.S)
        if not m:
            continue
        rows = [ln for ln in m.group(1).split("\\\\") if ln.strip()]
        if rows and len(rows) == int(size[0]):
            out.append(p.question)
            if len(out) >= limit:
                break
    return out


def _benchmark_prompt(subcat, size):
    """One of the benchmark's own prompts for a subcategory and size."""
    path = os.path.join(CACHE, f"linalg_bench_{size}.csv")
    if not os.path.exists(path):
        return ""
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["Subcat"] == subcat:
                return row["problem_latex"]
    return ""


def _corpus_prompt_for(family, size):
    """One of the corpus's prompts at a given size."""
    prompts = _corpus_prompts(family, size, limit=1)
    return prompts[0] if prompts else None


if __name__ == "__main__":
    unittest.main()
