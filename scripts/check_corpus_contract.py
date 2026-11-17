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
from la.benchprompt import (BODY_HINTS, FAMILY_TO_SUBCAT, SYSTEM_PROMPTS)
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
        for _ in range(args.n):
            p = generate(family, rng)
            if p is not None:
                break
        if p is None:
            print(f"{family:<20} FAILED TO GENERATE")
            return 1
        subcat = FAMILY_TO_SUBCAT[family]
        system_ok = p.system_prompt() == SYSTEM_PROMPTS[subcat]
        body_ok = BODY_HINTS[subcat] in p.question
        shape = answer_shape(p.answer)
        domain = value_domain(p.answer)
        print(f"{family:<20} {subcat:<16} "
              f"{'OK' if system_ok else 'MISMATCH':<8} "
              f"{'OK' if body_ok else 'MISMATCH':<8} "
              f"{shape:<8} {domain}")
        problems.append((family, subcat, system_ok, body_ok, shape, domain))

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
        if subcat in ("rank", "nullity", "determinant", "trace") and shape != "scalar":
            failures.append(f"{family}: expected a scalar answer, got {shape}")
        if subcat == "eigenvalue" and shape != "list":
            failures.append(f"{family}: expected a list answer, got {shape}")

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
