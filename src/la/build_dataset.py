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


def _config_weights() -> dict[str, int]:
    """The family weights declared in configs/default.yaml."""
    import pathlib
    import yaml
    path = pathlib.Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return dict((config.get("data") or {}).get("family_weights") or {})


def _rng_for(seed: int, index: int) -> random.Random:
    """The random stream for one corpus position.

    Deterministic in (seed, index) and INDEPENDENT of everything else.
    That independence is load-bearing: the builder interleaves the
    families, so if one generator consumed a different number of draws
    from its stream, every matrix after it in EVERY family would change.
    Run 8 changed `draw_entry()` from one draw per entry to two, and
    that silently rewrote 76% of the corpus - which made the run's
    result unreadable.

    Keep this in one place. A generator that reads from a shared stream
    is a bug that no test on the output can see.
    """
    return random.Random((seed * 1_000_003 + index) & 0xFFFFFFFF)


def _make_one(args):
    """Build one problem. Deterministic in (seed, index)."""
    seed, index, families, schedule = args
    fam = schedule[index % len(schedule)] if schedule else families[index % len(families)]
    rng = _rng_for(seed, index)
    p = generate(fam, rng)
    return (index, p)


def _weighted_schedule(families: list[str], weights: dict[str, int] | None) -> list[str]:
    """A deterministic schedule of families in the declared proportions.

    The schedule is one cycle long and the families are interleaved by
    their declared share, so every window of the schedule has the same mix
    rather than repeating one family thousands of times. That matters: a
    corpus built from a blocked schedule starts with a long run of one
    family, and a small build would contain none of the others.
    """
    if not weights:
        return list(families)
    chosen = [f for f in families if f in weights and weights[f] > 0]
    if not chosen:
        return list(families)
    total = sum(weights[f] for f in chosen)
    slots = 10_000
    counts = {f: max(1, round(weights[f] / total * slots)) for f in chosen}
    schedule = []
    for f in chosen:
        schedule.extend([f] * counts[f])
    # interleave proportionally: repeatedly take the family with the most
    # remaining slots, weighted by what it is owed
    remaining = dict(counts)
    ordered = []
    while any(v > 0 for v in remaining.values()):
        best = max(chosen, key=lambda f: (remaining[f] / counts[f], -chosen.index(f)))
        ordered.append(best)
        remaining[best] -= 1
    return ordered


def build(n: int, seed: int, families: list[str], out_path: str,
          render_first: int = 0, render_dir: str | None = None,
          workers: int = 0,
          family_weights: dict[str, int] | None = None) -> list[Problem]:
    """Generate n problems, in parallel if workers > 1.

    Every problem is a deterministic function of (seed, index), so the corpus
    is identical whatever order the work happens in and whatever the worker
    count. That is what makes a parallel build safe.
    """
    t0 = time.time()
    problems: list[Problem] = []
    schedule = _weighted_schedule(families, family_weights)
    if workers and workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            it = pool.imap_unordered(_make_one,
                                     ((seed, i, families, schedule)
                                      for i in range(n * 3)),
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
            fam = schedule[len(problems) % len(schedule)]
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

    family_weights = _config_weights()

    os.makedirs(args.out, exist_ok=True)
    print(f"generating {args.n_train:,} training problems")
    train = build(args.n_train, args.seed, families,
                  os.path.join(args.out, "train.jsonl"),
                  render_first=args.render_first,
                  render_dir=os.path.join(args.out, "figures"),
                  workers=args.workers, family_weights=family_weights)

    print(f"generating {args.n_test} held-out problems")
    # a different seed stream, and a different family order, so the test set
    # is not a prefix of the training set
    test = build(args.n_test, args.seed + 100003, list(reversed(families)),
                 os.path.join(args.out, "test.jsonl"), workers=args.workers,
                 family_weights=family_weights)

    counts = Counter(p.family for p in train)
    print("\nfamily balance in the training set:")
    for fam in families:
        print(f"  {fam:20} {counts.get(fam, 0):>7,}")
    print(f"\ntrain {len(train):,}   test {len(test):,}   out {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
