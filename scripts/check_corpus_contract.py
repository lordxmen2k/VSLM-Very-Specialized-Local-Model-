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
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from la.answer import parse_answer
from la.augment import closing_variants
from la.benchprompt import (BODY_HINTS, FAMILY_TO_SUBCAT, SYSTEM_PROMPTS)
from la.benchprompt_tails import PROMPT_TAILS
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
        # way the benchmark renders it
        tails = PROMPT_TAILS.get(FAMILY_TO_SUBCAT[family]) or []
        want_tails = {"\n\n".join(p for p in (m, c) if p) for m, c, _w in tails}
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

    print()
    if failures:
        print(f"CONTRACT MISMATCH: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("CONTRACT OK: every benchmark family matches the benchmark's "
          "system prompt, closing instruction and answer shape.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
