"""Explain what the model actually wrote, for the questions it got wrong.

`make eval` prints the score and a few failures, but a score says how many
were wrong, not why. The question that matters after a training run is
whether the model wrote a derivation at all: a model that emits a
derivation and gets the arithmetic wrong is a different problem from a model
that jumps straight to the answer.

    python -m la.explain data/results.jsonl
    python -m la.explain data/results.jsonl --family matrix_multiply --n 3

The file is the one `make eval` writes: one JSON object per example, with
the raw completion in the "raw" field.
"""
from __future__ import annotations

import argparse
import json
import sys


def _split(raw: str) -> dict:
    """Where the model's answer block starts, and how long the scratchpad was."""
    out = {"has_scratchpad": False, "scratchpad_lines": 0, "steps": 0,
           "has_answer": False, "has_plot": False}
    if not raw:
        return out
    out["has_scratchpad"] = "### Scratchpad" in raw
    out["has_answer"] = "### Answer" in raw
    out["has_plot"] = "### Plot" in raw
    lines = raw.splitlines()
    out["scratchpad_lines"] = len(lines)
    out["steps"] = sum(1 for l in lines if l.strip().startswith("STEP"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="data/results.jsonl")
    ap.add_argument("--family", default="", help="only this family")
    ap.add_argument("--n", type=int, default=3, help="how many to show per family")
    ap.add_argument("--lines", type=int, default=8,
                    help="how many lines of each completion to print")
    ap.add_argument("--all", action="store_true", help="show every failure")
    args = ap.parse_args(argv)

    try:
        with open(args.results, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
    except FileNotFoundError:
        print(f"no such file: {args.results}\nrun `make eval` first.")
        return 1

    failures = [r for r in records if not r.get("correct")]
    print(f"{len(records)} records, {len(failures)} wrong\n")

    # the headline diagnostic
    no_scratch = [r for r in failures if not _split(r.get("raw", ""))["has_scratchpad"]]
    no_answer = [r for r in failures if not _split(r.get("raw", ""))["has_answer"]]
    print(f"  of the failures, {len(no_scratch)} wrote no scratchpad at all "
          f"({100*len(no_scratch)/max(len(failures),1):.0f}%)")
    print(f"  and {len(no_answer)} wrote no answer block "
          f"({100*len(no_answer)/max(len(failures),1):.0f}%)")
    steps = [_split(r.get("raw", ""))["steps"] for r in failures]
    if steps:
        print(f"  the failures wrote a median of {sorted(steps)[len(steps)//2]} "
              f"STEP lines\n")

    by_family: dict[str, list[dict]] = {}
    for r in failures:
        by_family.setdefault(r.get("family", "?"), []).append(r)

    print(f"{'family':<22}{'wrong':>6}{'no scratchpad':>15}{'median steps':>14}")
    for family, rows in sorted(by_family.items()):
        n = len(rows)
        ns = sum(1 for r in rows
                 if not _split(r.get("raw", ""))["has_scratchpad"])
        st = sorted(_split(r.get("raw", ""))["steps"] for r in rows)
        print(f"  {family:<20}{n:>6}{ns:>15}{st[len(st)//2]:>14}")

    shown = 0
    for family, rows in sorted(by_family.items()):
        if args.family and family != args.family:
            continue
        limit = len(rows) if args.all else min(args.n, len(rows))
        for r in rows[:limit]:
            shown += 1
            print("\n" + "=" * 70)
            print(f"  {family}   {r.get('reason','')}")
            print("=" * 70)
            print("  Q:", (r.get("question") or "")[:200].replace("\n", " "))
            print()
            for line in (r.get("raw") or "").splitlines()[: args.lines]:
                print("   ", line[:160])
            if len((r.get("raw") or "").splitlines()) > args.lines:
                print("    ...")
            print()
            print("  A:", (r.get("answer") or "")[:160].replace("\n", " "))
    if not shown and args.family:
        print(f"\nno failures in {args.family}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
