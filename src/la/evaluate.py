"""Evaluate the model on the held-out set.

Four numbers matter, and only one of them is the headline:

    answer accuracy       the graded mathematics. This is the number.
    plot kind accuracy    did it choose the right kind of figure
    plot value accuracy   did it put the right numbers in it
    abstention rate       on problems it cannot do, did it say so

The last one matters more than it looks. A model that always answers has a
100 % answer rate on the problems it guessed and a 0 % rate on the ones it did
not, and you cannot tell the difference from the headline number alone. A
model that abstains correctly on a fifth of the problems is more useful than
one that confabulates the same fraction, and the report has to show both.

    python -m la.evaluate --data data/test.jsonl --adapter runs/vslm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .answer import check_answer
from .grading import extract_model_answer

# Six significant figures is what the corpus states for the rounded
# families, so the comparison has to be at least that loose. An answer that
# is off by one in an integer is still rejected.
EVAL_TOL = 1e-4
from .inference import generate, load_model
from .parse import parse_completion, summarise_answer
from .schema import features_match, spec_features


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the model")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--adapter", default="runs/vslm")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="", help="write the per-example results here")
    ap.add_argument("--show", type=int, default=8, help="how many failures to print")
    ap.add_argument("--prompt-mode", choices=["benchmark", "training"],
                    default="benchmark",
                    help="benchmark: the bare question, which is what the "
                         "benchmark asks. training: the wrapper the third "
                         "run was trained on, which measures the model in "
                         "distribution. The difference between the two is "
                         "the cost of the prompt mismatch")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} held-out problems from {args.data} "
          f"(prompt: {args.prompt_mode})")

    tok, model = load_model(args.base, args.adapter if os.path.isdir(args.adapter) else None)
    t0 = time.time()
    if args.prompt_mode == "training":
        prompts = [f"### Problem\n{r['question']}\n\n### Scratchpad\n"
                   for r in rows]
    else:
        prompts = [r["question"] for r in rows]
    outs = generate(tok, model, prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, batch_size=args.batch_size)
    print(f"generation took {(time.time()-t0)/60:.1f} min")

    n = len(rows)
    ans_ok = 0
    kind_ok = 0
    plot_ok = 0
    abstain = 0
    unparseable = 0
    per_family: dict[str, list[int]] = {}
    records = []
    failures = []

    for row, raw in zip(rows, outs):
        parsed = parse_completion(raw)
        mode = row.get("compare_mode", "exact")
        # A tolerance, not exact equality: some families state a value
        # rounded to six significant figures (the singular values, and the
        # condition number built from them), so a model that reproduces the
        # same six figures would otherwise be marked wrong.
        # The same tolerant extraction the benchmark uses: the answer block
        # if the model wrote one, otherwise the last \boxed{}, otherwise the
        # last line that looks like an answer. A model that answers in prose
        # with a boxed answer is answering, and the grader has to read it.
        answer = extract_model_answer(raw)
        res = check_answer(answer, row["truth"], mode=mode, tol=EVAL_TOL)
        a_ok = bool(res["correct"])
        ans_ok += a_ok

        ref = row.get("spec") or {}
        got = parsed.get("spec")
        g_kind = bool(got and got.get("kind") == ref.get("kind"))
        g_val = bool(got and ref and features_match(spec_features(got), spec_features(ref)))
        kind_ok += g_kind
        plot_ok += g_val

        said_no = ("cannot" in answer.lower()
                   or "not enough information" in answer.lower()
                   or "insufficient" in answer.lower())
        abstain += said_no
        if not answer:
            unparseable += 1

        fam = row["family"]
        per_family.setdefault(fam, []).append(int(a_ok))
        rec = {"family": fam, "question": row["question"], "raw": raw,
               "answer": answer, "correct": a_ok,
               "reason": res["reason"], "plot_kind_ok": g_kind,
               "plot_value_ok": g_val, "abstained": said_no}
        records.append(rec)
        if not a_ok and len(failures) < args.show:
            failures.append(rec)

    print("\n" + "=" * 64)
    print(f"  answer correct        {ans_ok}/{n}  ({100*ans_ok/max(n,1):.1f}%)")
    print(f"  plot kind correct     {kind_ok}/{n}  ({100*kind_ok/max(n,1):.1f}%)")
    print(f"  plot values correct   {plot_ok}/{n}  ({100*plot_ok/max(n,1):.1f}%)")
    print(f"  abstained             {abstain}/{n}  ({100*abstain/max(n,1):.1f}%)")
    print(f"  no answer produced    {unparseable}/{n}")
    print("=" * 64)
    print("\nby family:")
    for fam in sorted(per_family):
        v = per_family[fam]
        print(f"  {fam:20} {sum(v):>4}/{len(v):<4} ({100*sum(v)/max(len(v),1):5.1f}%)")
    print("\nby difficulty:")
    by_diff: dict[int, list[int]] = {}
    for row, rec in zip(rows, records):
        by_diff.setdefault(row.get("difficulty", 2), []).append(int(rec["correct"]))
    for d in sorted(by_diff):
        v = by_diff[d]
        print(f"  difficulty {d}         {sum(v):>4}/{len(v):<4} ({100*sum(v)/max(len(v),1):5.1f}%)")

    if failures:
        print(f"\n{args.show} failures:")
        for rec in failures:
            print("-" * 60)
            print("  Q:", rec["question"][:160])
            print("  A:", summarise_answer(rec["answer"]))
            print("  reason:", rec["reason"], "| plot kind ok:", rec["plot_kind_ok"])

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nwrote per-example results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
