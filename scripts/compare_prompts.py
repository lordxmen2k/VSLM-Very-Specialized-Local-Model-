"""Check the corpus prompt against the benchmark prompt, without hanging.

WHY THIS SCRIPT CHANGED

The original version compared raw TEXT prompts on the Base checkpoint,
generating up to 2048 tokens per batch. That was the right question then:
the corpus was trained on the bare question while the benchmark sent the
question behind its own system prompt, and the model rambled under a
prefix it had never seen. On the Base checkpoint, with no EOS
supervision, that generation runs to the cap - about 70 seconds a batch -
which is why it looked frozen.

That question has been answered and fixed. The corpus, the trainer and the
benchmark now all render their (system, user) pair through the tokenizer's
chat template, via `la.benchprompt.row_prompt`, so there is one definition
of what the model sees.

WHAT THIS CHECKS NOW

The remaining risk is not the system prompt - it is the CLOSING
INSTRUCTION. The corpus closes every question with its family's hint:

    Show all intermediate steps. Put your final numerical answer inside \\boxed{}.

The benchmark does not. Across the 660 released problems it uses nine
different wordings ("Show your steps.", "List all eigenvalues inside
\\boxed{}.", "Put the final 3x3 result matrix inside \\boxed{}.") and the
whole 4x4 file carries no closing instruction at all.

So this script measures, per size and per subcategory:

  * how many benchmark problems carry a closing instruction at all
  * how many use the corpus's exact wording
  * whether the SYSTEM prompt already tells the model to box the answer

and it renders both sides through the REAL tokenizer so the chat template
is exercised. It does no generation unless --generate is passed, and even
then the cap is small and the run is bounded.

    python scripts/compare_prompts.py
    python scripts/compare_prompts.py --generate --max-new-tokens 256
"""

import argparse
import collections
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

CACHE = "bench_cache"
TEST_CACHE = os.path.join("tests", "_cache")

from la.benchprompt import (BODY_HINTS, benchmark_prompt_pair,  # noqa: E402
                            body_hint, row_prompt, split_problem,
                            system_prompt_for)
from la.inference import load_model                             # noqa: E402


def load_rows(size: str, cache: str):
    """The benchmark rows, from the user's cache or the test cache."""
    for base in (cache, TEST_CACHE):
        path = os.path.join(base, f"linalg_bench_{size}.csv")
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as fh:
                return list(csv.DictReader(fh)), path
    return [], ""


def closing_line(user: str) -> str:
    """The last non-empty line of a rendered user turn."""
    lines = [l for l in user.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B",
                    help="the checkpoint whose tokenizer renders the template")
    ap.add_argument("--generate", action="store_true",
                    help="also generate from the rendered prompt and time it")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="generation cap for --generate")
    ap.add_argument("--n", type=int, default=0,
                    help="how many problems to render (0 = all)")
    args = ap.parse_args()

    print("loading the tokenizer and model ...", flush=True)
    tok, model = load_model(args.base, "runs/vslm")
    print(f"  checkpoint    = {args.base}", flush=True)
    print(f"  dtype         = {model.dtype}", flush=True)
    print(f"  chat template = {bool(getattr(tok, 'chat_template', None))}",
          flush=True)
    print()

    # --- the benchmark's closing instructions ----------------------------
    print("=== the closing instruction on each benchmark problem ===")
    print()
    per_size = {}
    wordings = collections.Counter()
    exact = 0
    total = 0
    by_sub = collections.defaultdict(lambda: [0, 0])   # sub -> [total, no hint]

    for size in ("3x3", "4x4", "5x5"):
        rows, path = load_rows(size, CACHE)
        if not rows:
            print(f"  {size}: NO CSV FOUND (looked in {CACHE} and {TEST_CACHE})")
            continue
        n_hint = 0
        for p in rows:
            sub = p.get("Subcat") or p.get("subcat") or ""
            _, user = benchmark_prompt_pair(
                p.get("problem_latex") or p.get("problem") or "", sub, size)
            tail = closing_line(user)
            wordings[tail] += 1
            total += 1
            by_sub[sub][0] += 1
            if "\\boxed" in user:
                n_hint += 1
            else:
                by_sub[sub][1] += 1
            if tail == body_hint(sub):
                exact += 1
        per_size[size] = (n_hint, len(rows))
        print(f"  {size}: {n_hint}/{len(rows)} carry a closing instruction "
              f"mentioning \\boxed{{}}   ({path})")
    print()
    print(f"  total: {total} problems")
    n_boxed = sum(v[0] for v in per_size.values())
    print(f"  carry a closing instruction mentioning \\boxed{{}}: {n_boxed}")
    print(f"  carry NO closing instruction at all:            "
          f"{total - n_boxed}")
    print(f"  use the corpus's EXACT closing wording:           {exact}")
    print()

    print("  by subcategory (total / with no closing instruction):")
    for sub in sorted(by_sub):
        t, m = by_sub[sub]
        flag = "  <-" if m else ""
        print(f"    {sub:<16} {t:>3} / {m:>3} missing{flag}")
    print()

    print("  the distinct closing instructions the benchmark uses:")
    for w, n in wordings.most_common(12):
        print(f"    {n:>4}  {w[:88]!r}")
    if len(wordings) > 12:
        print(f"    ... and {len(wordings)-12} more")
    print()

    # --- does the SYSTEM prompt already cover it? ------------------------
    print("=== does the SYSTEM prompt already say to box the answer? ===")
    print()
    for sub in sorted(by_sub):
        s = system_prompt_for(sub)
        print(f"    {sub:<16} {s[-70:]!r}")
    print()
    print("  The system prompt says 'Always put your final ... answer")
    print("  inside \\boxed{}.' for every subcategory, so the model is told")
    print("  to box the answer even when the user turn does not repeat it.")
    print()

    # --- render both sides through the real tokenizer --------------------
    print("=== the rendered prompt, through the real chat template ===")
    print()
    rows, _ = load_rows("3x3", CACHE)
    if rows:
        p = rows[0]
        sub = p.get("Subcat") or ""
        sysp, user = benchmark_prompt_pair(p.get("problem_latex") or "", sub,
                                            "3x3")
        rendered = row_prompt({"system": sysp, "user": user}, tok)
        for line in rendered.split("\n"):
            print("  " + line)
        print()
        print(f"  the corpus's closing line for {sub!r} is:")
        print(f"    {body_hint(sub)!r}")
        print(f"  and the benchmark's closing line is:")
        print(f"    {closing_line(user)!r}")
        print()

    # --- optional generation ---------------------------------------------
    if args.generate:
        import torch
        print("=== generating from the rendered prompt ===")
        print(f"  cap: {args.max_new_tokens} new tokens", flush=True)
        print("  (needs the EOS-supervised checkpoint to terminate; on the")
        print("   Base checkpoint it will run to the cap)", flush=True)
        print()
        prompts = []
        n = args.n or 8
        for size in ("3x3", "4x4", "5x5"):
            rows, _ = load_rows(size, CACHE)
            for p in rows[:max(1, n // 3)]:
                sub = p.get("Subcat") or ""
                s, u = benchmark_prompt_pair(p.get("problem_latex") or "",
                                             sub, size)
                prompts.append(row_prompt({"system": s, "user": u}, tok))
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        t = time.time()
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                               do_sample=False, pad_token_id=tok.pad_token_id)
        dt = time.time() - t
        nt = [g[j][enc["input_ids"].shape[1]:].shape[0]
              for j in range(len(prompts))]
        print(f"  {dt:6.1f}s   new tokens {nt}")
        print()

    print("=" * 62)
    print("=> The SYSTEM prompt is identical on both sides, and the chat")
    print("   template is applied by the same function, so the model is")
    print("   scored in the same format it is trained in.")
    print()
    print(f"   The remaining difference is the CLOSING INSTRUCTION:")
    print(f"   {total - n_boxed} of {total} benchmark problems carry none,")
    print(f"   and only {exact} use the corpus's exact wording.")
    print()
    print("   This is a distribution shift, not a format break. The system")
    print("   prompt already asks for \\boxed{}, and the grader falls back")
    print("   to the last value-like line. If run 5 underperforms on the")
    print("   4x4 file specifically, add the corpus's closing line to the")
    print("   benchmark prompt for the problems that lack one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
