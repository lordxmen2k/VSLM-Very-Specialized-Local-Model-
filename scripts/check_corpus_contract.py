"""Print the corpus's answer contract, family by family.

The contract has four legs: the system prompt, the closing instruction,
the answer's shape, and the answer's value domain. A mismatch in any one
of them is invisible in the score and costs the whole family.

    python scripts/check_corpus_contract.py --n 200

Exits 0 when every benchmark family matches the benchmark's own contract,
and 1 when one does not.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from la.answer import parse_answer
from la.augment import closing_variants
from la.benchprompt import (BODY_HINTS, FAMILY_TO_SUBCAT, SYSTEM_PROMPTS)
from la.benchprompt_tails import PROMPT_TAILS, PROMPT_TAILS_BY_SIZE
from la.problems import GENERATORS, generate


BENCHMARK_SUBCATS = {
    "determinant", "eigenvalue", "rank", "nullity", "trace", "transpose",
    "multiplication", "matrix_power", "matrix_vector",
}


def answer_shape(answer: str) -> str:
    """The shape of a graded answer: scalar, list, set or dict."""
    parsed = parse_answer(answer)
    if isinstance(parsed, dict):
        return "pair"
    if isinstance(parsed, list):
        return "list"
    if isinstance(parsed, (tuple, set)):
        return "list"
    return "scalar"


def value_domain(answer: str) -> str:
    """The value domain of a graded answer."""
    parsed = parse_answer(answer)
    values = parsed if isinstance(parsed, list) else [parsed]
    if not values:
        return "empty"
    if all(getattr(v, "is_Integer", None) is not None for v in values):
        return "integer"
    if all("." in str(v) for v in values):
        return "decimal"
    if any(getattr(v, "is_number", False) for v in values):
        return "exact number"
    return "other"


def _check_task_statements(families, n):
    """The corpus's task statement must be the SCORER's, per size.

    The contract is what the model is actually sent at scoring time,
    which is NOT the raw problem file: the scorer reconstructs the 4x4
    task from the subcategory label, so the 4x4 file - a bare matrix
    with no task statement - is scored with "Find the determinant of the
    4×4 matrix A." in front of it.

    Comparing the corpus against the raw file therefore flags the
    CORRECT 4x4 statement as an error. The corpus used to invent its own
    wordings - 66 to 78 per cent of them naming no size at all - and to
    write the size with an ASCII "x" where the benchmark writes U+00D7.
    The scorer is the reference, and it is the reference the model is
    graded against.

    Run 10 scored 276/660 = 41.8% with the determinant at 1/150 against
    the base model's 54/150, and this is why.
    """
    failures = []
    print()
    print(f"{'family':<20} {'scorer task (3x3)':<36} {'corpus'}")
    for family in families:
        subcat = FAMILY_TO_SUBCAT[family]
        want = _scorer_task_statements(subcat)
        if not want:
            continue
        rng = random.Random(20261117)
        got = {}
        for _ in range(n):
            p = generate(family, rng)
            if p is None:
                continue
            parts = [x.strip() for x in p.question.split("\n\n")]
            body = parts[1] if len(parts) > 1 else ""
            task = body.split("\n\n")[0].strip() if body else ""
            got[task] = got.get(task, 0) + 1
        b3 = want.get("3x3", "")
        ctxt = " | ".join(sorted(t for t in got if t)[:2]) or "(none)"
        print(f"{family:<20} {b3[:34]:<36} {ctxt[:44]}")
        want_all = set()
        for text in want.values():
            want_all |= set(text.split(" | "))
        extra = set(t for t in got if t) - want_all
        if extra:
            failures.append(
                f"{family}: the corpus produces task statements the "
                f"scorer never sends: {sorted(extra)[:3]}")
        missing = want_all - set(got)
        if missing:
            failures.append(
                f"{family}: the corpus never produces the scorer's task "
                f"statement {sorted(missing)[:2]}")
    return failures


def _scorer_task_statements(subcat):
    """The task statement the SCORER sends, per size.

    Read off the scorer's own prompt, so the 4x4 reconstruction is
    included exactly as the model receives it.
    """
    import csv
    from la.benchprompt import benchmark_prompt_pair
    cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "tests", "_cache")
    out = {}
    for size in ("3x3", "4x4", "5x5"):
        path = os.path.join(cache, f"linalg_bench_{size}.csv")
        if not os.path.exists(path):
            continue
        texts = set()
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["Subcat"] != subcat:
                    continue
                _sys, user = benchmark_prompt_pair(
                    row["problem_latex"], subcat, size)
                pars = [p for p in user.split("\n\n") if p]
                if len(pars) > 1 and not pars[1].startswith("A ="):
                    texts.add(pars[1].strip())
        if texts:
            out[size] = " | ".join(sorted(texts))
    return out


def _benchmark_task_statements(subcat):
    """The task statement the benchmark uses, per size."""
    import csv
    cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "tests", "_cache")
    out = {}
    for size in ("3x3", "4x4", "5x5"):
        path = os.path.join(cache, f"linalg_bench_{size}.csv")
        if not os.path.exists(path):
            continue
        counts = Counter()
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["Subcat"] != subcat:
                    continue
                first = row["problem_latex"].split("\n\n")[0].strip()
                if first.startswith("A ="):
                    first = ""
                counts[first] += 1
        out[size] = counts
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200,
                    help="how many problems to sample per family")
    args = ap.parse_args(argv)

    families = [f for f in sorted(GENERATORS) if f in FAMILY_TO_SUBCAT]
    print(f"sampling {args.n} problems from each of {len(families)} "
          f"benchmark corpus families\n")
    header = (f"{'family':<20} {'subcat':<16} {'system':<8} {'body':<8} "
              f"{'shape':<8} {'domain'}")
    print(header)
    print("-" * len(header))

    problems = []
    for family in families:
        rng = random.Random(20261117)
        p = None
        # Check EVERY sampled problem, not just the first one. One example
        # in five has no closing instruction, so a checker that looks at a
        # single sample has a one-in-five chance of passing a family whose
        # other four-fifths carry the wrong instruction - and, worse, a
        # one-in-five chance of FAILING a family that is correct.
        bad_system = 0
        bad_body = 0
        bare = 0
        seen = 0
        shapes = set()
        # the closing instructions this family is allowed to use: the
        # benchmark's own set for its subcategory
        allowed = closing_variants(family)
        # every tail the benchmark uses for this subcategory, rendered the
        # way the benchmark renders it: the method and the closing
        # instruction are ONE paragraph separated by a single newline,
        # and the table is per SIZE.
        tails = PROMPT_TAILS.get(FAMILY_TO_SUBCAT[family]) or []
        want_tails = set()
        for _size, _entries in PROMPT_TAILS_BY_SIZE.get(
                FAMILY_TO_SUBCAT[family], {}).items():
            for m, c, _w in _entries:
                want_tails.add(f"{m}\n{c}" if m and c else (m or c))
        got_tails = set()
        for _ in range(args.n):
            p = generate(family, rng)
            if p is None:
                continue
            seen += 1
            subcat = FAMILY_TO_SUBCAT[family]
            if p.system_prompt() != SYSTEM_PROMPTS[subcat]:
                bad_system += 1
            if not any(v in p.question for v in allowed):
                # no closing instruction at all is the bare-matrix shape:
                # all 220 of the benchmark's 4x4 problems are a matrix with
                # no closing instruction, so it is a valid contract
                bare += 1
            # which benchmark tail does this problem's prompt carry?
            for tail in want_tails:
                if not tail:
                    # the bare-matrix shape: no closing instruction and no
                    # method instruction. It cannot be detected by
                    # "tail in question" because "" is in every string.
                    if not any(v in p.question for v in allowed):
                        got_tails.add(tail)
                    continue
                if tail in p.question:
                    got_tails.add(tail)
            shapes.add(answer_shape(p.answer))
        # every tail the benchmark uses must appear in the corpus
        missing = want_tails - got_tails
        if not seen:
            print(f"{family:<20} FAILED TO GENERATE")
            return 1
        subcat = FAMILY_TO_SUBCAT[family]
        system_ok = bad_system == 0
        body_ok = bad_body == 0 and not missing
        shape = "/".join(sorted(shapes))
        domain = value_domain(p.answer)
        print(f"{family:<20} {subcat:<16} "
              f"{'OK' if system_ok else 'MISMATCH':<8} "
              f"{'OK' if body_ok else 'MISMATCH':<8} "
              f"{shape:<8} {domain}"
              f"   (bare {bare}/{seen}, tails {len(got_tails)}/"
              f"{len(want_tails)})")
        problems.append((family, subcat, system_ok, body_ok, shape, domain))
        if not system_ok:
            print(f"  !! {bad_system} of {seen} had the wrong system prompt")
        if missing:
            for tail in sorted(missing):
                print(f"  !! the corpus never produces the benchmark's "
                      f"prompt tail: {tail!r}")

    failures = []
    for family, subcat, system_ok, body_ok, _shape, _domain in problems:
        if not system_ok:
            failures.append(f"{family}: system prompt is not the benchmark's "
                            f"{subcat} prompt")
        if not body_ok:
            failures.append(f"{family}: closing instruction is not the "
                            f"benchmark's {subcat} instruction")
    # the scalar and list shapes the grader can read
    for family, subcat, _s, _b, shape, _d in problems:
        shapes = set(shape.split("/"))
        if subcat in ("rank", "nullity", "determinant", "trace"):
            if shapes != {"scalar"}:
                failures.append(f"{family}: expected a scalar answer, got "
                                f"{'/'.join(sorted(shapes))}")
        if subcat == "eigenvalue" and shapes != {"list"}:
            failures.append(f"{family}: expected a list answer, got "
                            f"{'/'.join(sorted(shapes))}")

    # the SIZE MIX, which is the sixth leg of the contract
    size_failures = _check_size_mix(families, args.n)
    failures.extend(size_failures)

    # the TASK STATEMENT, which is the seventh leg
    task_failures = _check_task_statements(families, args.n)
    failures.extend(task_failures)

    print()
    if failures:
        print(f"CONTRACT MISMATCH: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("CONTRACT OK: every benchmark family matches the benchmark's "
          "system prompt, closing instruction, answer shape and size mix.")
    return 0


def _matrix_size(text):
    """The dimension of the first matrix in a problem's question."""
    m = re.search(r"\\begin\{[bp]matrix\}(.*?)\\end\{[bp]matrix\}", text, re.S)
    if not m:
        return None
    rows = [ln for ln in m.group(1).split("\\\\") if ln.strip()]
    return len(rows) if rows else None


def _benchmark_sizes(subcat):
    """The dimension of every matrix the benchmark scores for a subcat."""
    import csv
    cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "tests", "_cache")
    out = []
    for size in ("3x3", "4x4", "5x5"):
        path = os.path.join(cache, f"linalg_bench_{size}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["Subcat"] != subcat:
                    continue
                for m in re.finditer(
                        r"[A-Z](?:_\{[A-Z]+\})? = \\begin\{[bp]matrix\}(.*?)"
                        r"\\end\{[bp]matrix\}", row["problem_latex"], re.S):
                    body = m.group(1)
                    rows = [ln for ln in body.split("\\\\") if ln.strip()]
                    if rows:
                        out.append(len(rows))
    return out


def _check_size_mix(families, n):
    """The corpus's size mix must be the benchmark's.

    The benchmark's mix is exactly uniform for every family, and there is
    not one 2x2 matrix anywhere in all 660 problems. The corpus used to
    teach 2x2 - between 10 and 40 per cent of its rows - which are rows
    NOT spent on the 5x5 problems the benchmark does score.
    """
    failures = []
    print()
    print(f"{'family':<20} {'benchmark sizes':<22} {'corpus sizes':<22}")
    for family in families:
        subcat = FAMILY_TO_SUBCAT[family]
        bench = _benchmark_sizes(subcat)
        if not bench:
            continue
        rng = random.Random(20261117)
        got = []
        for _ in range(n):
            p = generate(family, rng)
            if p is None:
                continue
            s = _matrix_size(p.question)
            if s:
                got.append(s)
        if not got:
            continue
        b = sorted(Counter(bench).items())
        c = sorted(Counter(got).items())
        btxt = " ".join(f"{k}:{v}" for k, v in b)
        ctxt = " ".join(f"{k}:{v}" for k, v in c)
        print(f"{family:<20} {btxt:<22} {ctxt:<22}")
        off = [s for s in got if s not in set(bench)]
        if off:
            failures.append(f"{family}: {len(off)} of {len(got)} matrices are a "
                            f"size the benchmark never scores "
                            f"({sorted(set(off))})")
        missing = set(bench) - set(got)
        if missing:
            failures.append(f"{family}: the corpus never generates a "
                            f"{sorted(missing)} matrix, but the benchmark "
                            f"scores them")
    return failures


if __name__ == "__main__":
    sys.exit(main())

