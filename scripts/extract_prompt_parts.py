#!/usr/bin/env python3
"""Extract the benchmark's task statements and prompt tails from the
released problem files.

The corpus's prompt contract - the task statement before the matrix and
the tail after it - used to be hand-written from the files' documentation.
That is how two defects survived:

  - the task statements were written with an ASCII "x" where the
    benchmark writes U+00D7 ("3x3" against "3×3"), and
  - the tail table mixed the sizes together, so the transpose's 3x3 tail
    could turn up in a 5x5 problem.

Both are invisible by eye and invisible to a test written against the
same hand-written table. So the tables are EXTRACTED here, from the
released files, and the tests check the tables against the files in both
directions.

    python scripts/extract_prompt_parts.py

prints the extracted tables. With --check it verifies that the tables in
src/la/benchprompt_tasks.py and src/la/benchprompt_tails.py still match
the files, and exits non-zero if they do not.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(REPO, "tests", "_cache")
SIZES = ("3x3", "4x4", "5x5")

SUBCATS = ("determinant", "eigenvalue", "matrix_power", "matrix_vector",
           "multiplication", "nullity", "rank", "trace", "transpose")


def _is_matrix(par):
    return "\\begin{" in par


def _task_paragraph(prompt):
    """The benchmark's task statement, or "" for the bare-matrix shape."""
    first = prompt.split("\n\n")[0].strip()
    if _is_matrix(first):
        return ""
    return first


def _tail_paragraphs(prompt):
    """The paragraphs after the matrix, as one string."""
    pars = [p for p in prompt.split("\n\n") if p]
    # drop the task paragraph and every matrix paragraph
    body = [p for p in pars if not _is_matrix(p)]
    if pars and not _is_matrix(pars[0]):
        body = body[1:]
    return "\n".join(body)


def extract():
    """The (tasks, tails) tables, from the released files."""
    tasks = {}
    tails = {}
    for subcat in SUBCATS:
        tasks[subcat] = {}
        tails[subcat] = {}
        for size in SIZES:
            path = os.path.join(CACHE, f"linalg_bench_{size}.csv")
            if not os.path.exists(path):
                continue
            task_counts = {}
            tail_counts = {}
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row["Subcat"] != subcat:
                        continue
                    task = _task_paragraph(row["problem_latex"])
                    tail = _tail_paragraphs(row["problem_latex"])
                    task_counts[task] = task_counts.get(task, 0) + 1
                    tail_counts[tail] = tail_counts.get(tail, 0) + 1
            if task_counts:
                tasks[subcat][size] = task_counts
            if tail_counts:
                tails[subcat][size] = tail_counts
    return tasks, tails


def check():
    """Do the tables in src/ still match the released files?"""
    sys.path.insert(0, os.path.join(REPO, "src"))
    from la.benchprompt_tasks import BENCHMARK_TASK_STATEMENTS
    from la.benchprompt_tails import PROMPT_TAILS_BY_SIZE

    tasks, tails = extract()
    problems = []

    for subcat, per_size in tasks.items():
        table = BENCHMARK_TASK_STATEMENTS.get(subcat, {})
        for size, counts in per_size.items():
            got = table.get(size)
            if got is None:
                problems.append(f"{subcat} {size}: no entry in "
                                f"BENCHMARK_TASK_STATEMENTS")
                continue
            got_map = {}
            for text, weight in got:
                got_map[text] = got_map.get(text, 0) + weight
            if got_map != counts:
                problems.append(
                    f"{subcat} {size}: the task table says {got_map} and the "
                    f"files say {counts}")

    for subcat, per_size in tails.items():
        table = PROMPT_TAILS_BY_SIZE.get(subcat, {})
        for size, counts in per_size.items():
            # the tail table stores (method, closing); the files give one
            # string, so split it back on the FIRST newline
            got = table.get(size)
            if got is None:
                problems.append(f"{subcat} {size}: no entry in "
                                f"PROMPT_TAILS_BY_SIZE")
                continue
            got_map = {}
            for method, closing, weight in got:
                text = f"{method}\n{closing}" if method and closing else (
                    method or closing)
                got_map[text] = got_map.get(text, 0) + weight
            if got_map != counts:
                problems.append(
                    f"{subcat} {size}: the tail table says {got_map} and the "
                    f"files say {counts}")

    if problems:
        print(f"PROMPT TABLES ARE STALE: {len(problems)} mismatch(es)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("PROMPT TABLES MATCH THE RELEASED FILES")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify the tables in src/ against the files")
    args = ap.parse_args(argv)

    if args.check:
        return check()

    tasks, tails = extract()
    print("TASK STATEMENTS")
    for subcat, per_size in tasks.items():
        print(f"  {subcat}")
        for size, counts in per_size.items():
            for text, n in counts.items():
                print(f"    {size}: {n:>3}x {text!r}")
    print()
    print("PROMPT TAILS")
    for subcat, per_size in tails.items():
        print(f"  {subcat}")
        for size, counts in per_size.items():
            for text, n in counts.items():
                print(f"    {size}: {n:>3}x {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
