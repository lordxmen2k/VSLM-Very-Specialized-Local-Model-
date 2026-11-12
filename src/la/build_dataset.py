"""Build the training corpus.

Everything is generated, so the corpus is reproducible from a seed and can be
rebuilt at any time. The held-out set is generated from a different seed and is
never used for training.

    python -m la.build_dataset --n-train 50000 --n-test 200 --seed 7 --out data
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

from .problems import GENERATORS, Problem, generate


def _make_one(args):
    """Build one problem. Deterministic in (seed, index)."""
    seed, index, families = args
    fam = families[index % len(families)]
    rng = random.Random((seed * 1_000_003 + index) & 0xFFFFFFFF)
    p = generate(fam, rng)
    return (index, p)


def build(n: int, seed: int, families: list[str], out_path: str,
          render_first: int = 0, render_dir: str | None = None,
          workers: int = 0) -> list[Problem]:
    """Generate n problems, in parallel if workers > 1.

    Every problem is a deterministic function of (seed, index), so the corpus
    is identical whatever order the work happens in and whatever the worker
    count. That is what makes a parallel build safe.
    """
    t0 = time.time()
    problems: list[Problem] = []
    if workers and workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            it = pool.imap_unordered(_make_one,
                                     ((seed, i, families) for i in range(n * 3)),
                                     chunksize=64)
            for idx, p in it:
                if p is None:
                    continue
                problems.append((idx, p))
                if len(problems) % 5000 == 0:
                    rate = len(problems) / (time.time() - t0)
                    print(f"  {len(problems):>7,} / {n:,}  ({rate:.0f}/s)", flush=True)
                if len(problems) >= n:
                    break
        pool.terminate()
        # imap_unordered returns in completion order; sort so the file is
        # identical whatever the worker count or the scheduling
        problems.sort(key=lambda ip: ip[0])
    else:
        rng = random.Random(seed)
        while len(problems) < n:
            fam = families[len(problems) % len(families)]
            p = generate(fam, rng)
            if p is None:
                continue
            problems.append((len(problems), p))
            if len(problems) % 5000 == 0:
                rate = len(problems) / (time.time() - t0)
                print(f"  {len(problems):>7,} / {n:,}  ({rate:.0f}/s)", flush=True)
    problems = [p for _i, p in problems][:n]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for p in problems:
            fh.write(json.dumps(p.to_json(), ensure_ascii=False) + "\n")
    print(f"  wrote {len(problems):,} problems to {out_path}")

    if render_first and render_dir:
        from .render import render
        from .schema import validate_spec
        os.makedirs(render_dir, exist_ok=True)
        drawn = 0
        for p in problems[:render_first]:
            if not p.spec:
                continue
            errs = validate_spec(p.spec)
            if errs:
                print(f"  spec invalid for {p.family}: {errs}")
                continue
            render(p.spec, os.path.join(render_dir, f"{p.family}_{drawn:04d}.png"))
            drawn += 1
        print(f"  drew {drawn} figures into {render_dir}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the linear algebra corpus")
    ap.add_argument("--n-train", type=int, default=50000)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data")
    ap.add_argument("--families", default=",".join(GENERATORS),
                    help="comma separated list of families")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel worker processes; 0 uses one core")
    ap.add_argument("--render-first", type=int, default=40,
                    help="how many training figures to render as a sanity check")
    args = ap.parse_args(argv)

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in GENERATORS]
    if unknown:
        print(f"unknown families: {unknown}", file=sys.stderr)
        print(f"available: {sorted(GENERATORS)}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    print(f"generating {args.n_train:,} training problems")
    train = build(args.n_train, args.seed, families,
                  os.path.join(args.out, "train.jsonl"),
                  render_first=args.render_first,
                  render_dir=os.path.join(args.out, "figures"),
                  workers=args.workers)

    print(f"generating {args.n_test} held-out problems")
    # a different seed stream, and a different family order, so the test set
    # is not a prefix of the training set
    test = build(args.n_test, args.seed + 100003, list(reversed(families)),
                 os.path.join(args.out, "test.jsonl"), workers=args.workers)

    counts = Counter(p.family for p in train)
    print("\nfamily balance in the training set:")
    for fam in families:
        print(f"  {fam:20} {counts.get(fam, 0):>7,}")
    print(f"\ntrain {len(train):,}   test {len(test):,}   out {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
