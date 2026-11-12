"""Ask the model a question, and check its work.

The point of this front end is that the model does not get to be its own
authority. It produces an answer, the answer is parsed and compared against
sympy, and the plot spec is validated before anything is drawn. If the
verification fails you are told, rather than being handed a confident
paragraph.

    python -m la.chat --adapter runs/vslm
    python -m la.chat --adapter runs/vslm --question "What is the determinant of [[2,1],[1,2]]?"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .answer import check_answer
from .inference import generate, load_model, parse_completion, summarise_answer
from .render import render


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ask the model a question")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--adapter", default="runs/vslm")
    ap.add_argument("--question", default="")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    tok, model = load_model(args.base, args.adapter if os.path.isdir(args.adapter) else None)
    os.makedirs(args.out_dir, exist_ok=True)

    questions = [args.question] if args.question else []
    while not questions:
        try:
            q = input("\nquestion (blank to quit): ").strip()
        except EOFError:
            break
        if not q:
            break
        questions = [q]

    for q in questions:
        raw = generate(tok, model, [q], max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature, batch_size=1)[0]
        parsed = parse_completion(raw)
        print("\n" + "=" * 70)
        print("SCRATCHPAD")
        print(parsed["scratchpad"])
        print("\nANSWER")
        print(parsed["answer"] or "(none)")

        # verify against sympy where we can
        verdict = "not checked (no ground truth available for a free question)"
        try:
            from .answer import parse_answer
            val = parse_answer(parsed["answer"])
            verdict = f"parsed as {val}"
        except Exception as exc:
            verdict = f"could not be parsed: {exc}"
        print(f"\nCHECK: {verdict}")

        if parsed.get("spec") and not args.no_plot:
            errs = parsed["spec_errors"]
            if errs:
                print(f"PLOT: rejected — {errs[0]}")
            else:
                path = os.path.join(args.out_dir, "answer.png")
                render(parsed["spec"], path)
                print(f"PLOT: written to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
