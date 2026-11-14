"""Benchmark the model against published results, and measure its speed.

Two things a local model needs and never gets told:

**Where it stands.** LinAlg-Bench is a purpose-built benchmark for exactly
this subject: SymPy-verified linear algebra problems across 3x3, 4x4 and 5x5
matrices and nine operation types, with ten frontier models scored on it. The
published numbers are in `PUBLISHED_LINALG` below, so the report prints your
score next to theirs.

The finding that matters is in that table: accuracy falls off a cliff between
4x4 and 5x5 — 32 percentage points on determinants — and the models that fail
there do not compute badly, they *stop computing* and fabricate. The corpus
here is built to attack that: 5x5 problems are in the training mix, every
derivation ends with a check, and the answer is graded by sympy rather than
believed.

**How fast it is.** `--perf` measures time to first token, decode throughput
at several batch sizes, peak VRAM and load time, at 4-bit, 8-bit and bf16.

    python -m la.bench --benchmark linalg --adapter runs/vslm
    python -m la.bench --benchmark linalg --adapter runs/vslm --baseline
    python -m la.bench --perf --adapter runs/vslm
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

from .answer import answers_equal, parse_answer
from .benchprompt import build_benchmark_prompt, has_task
from .grading import extract_model_answer, normalise_reference
from .inference import generate, load_model, parse_completion

# Accuracy on LinAlg-Bench by matrix size, as published in the benchmark's
# own results (mst-ai/linalg-bench-llm, results/accuracy/). These are the
# numbers a local 1.7B model should be measured against.
# LinAlg-Bench writes the eigenvalues rounded to four decimal places, so the
# reference and the true value differ by up to 5e-5. Comparing at the default
# tolerance of 1e-9 would score a correct eigenvalue wrong. Measured over the
# 660 problems: the largest gap for the eigenvalues is 4.991e-05, and for
# every other task it is exactly zero.
BENCH_TOL = 1e-3

PUBLISHED_LINALG = {
    "OpenAI-o1": (94.6, 92.3, 81.4),
    "Gemini-3.1-Pro": (93.7, 94.6, 88.0),
    "DeepSeek-V3": (89.3, 79.9, 62.5),
    "GPT-5.2": (85.8, 82.1, 69.2),
    "Mistral-Large": (81.4, 75.6, 45.1),
    "Qwen3-235B": (79.8, 80.5, 71.5),
    "Claude-4.5-Sonnet": (80.1, 69.7, 33.1),
    "Qwen2.5-72B": (69.4, 39.8, 24.6),
    "Llama-3.3-70B": (63.1, 60.6, 28.3),
    "GPT-4o": (66.6, 36.0, 28.7),
}

# Where the benchmark's problems live. The GitHub copy is plain CSV with no
# dependency on the HuggingFace libraries.
LINALG_SOURCES = [
    ("3x3", "https://raw.githubusercontent.com/linalgbench2026/LinAlgBench/"
            "main/data/linalg_bench_3x3.csv"),
    ("4x4", "https://raw.githubusercontent.com/linalgbench2026/LinAlgBench/"
            "main/data/linalg_bench_4x4.csv"),
    ("5x5", "https://raw.githubusercontent.com/linalgbench2026/LinAlgBench/"
            "main/data/linalg_bench_5x5.csv"),
]

LINALG_COLUMNS = {
    "problem": ("problem_latex", "problem", "Problem"),
    "answer": ("answer_latex", "answer", "Answer"),
    "id": ("Problem_ID", "problem_id", "id"),
    "task": ("Subcat", "subcat", "task_type"),
}


def _pick(row: dict, key: str) -> str:
    for name in LINALG_COLUMNS[key]:
        if name in row and row[name]:
            return str(row[name])
    return ""


def fetch_linalg(cache_dir: str = "bench_cache") -> list[dict]:
    """Download the benchmark problems and normalise them."""
    os.makedirs(cache_dir, exist_ok=True)
    problems: list[dict] = []
    for size, url in LINALG_SOURCES:
        path = os.path.join(cache_dir, f"linalg_bench_{size}.csv")
        if not os.path.exists(path):
            print(f"  downloading {size} problems ...", flush=True)
            with urllib.request.urlopen(url, timeout=120) as resp:
                data = resp.read()
            with open(path, "wb") as fh:
                fh.write(data)
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                q = _pick(row, "problem")
                a = _pick(row, "answer")
                if not q or not a:
                    continue
                subcat = _pick(row, "task")
                problems.append({"size": size, "question": q, "answer": a,
                                 "id": _pick(row, "id"), "task": subcat,
                                 "instructed": has_task(q),
                                 "prompt": build_benchmark_prompt(q, subcat, size)})
    return problems


def score(problems: list[dict], outs: list[str]) -> dict:
    """Grade against the benchmark's own ground truth, using sympy."""
    per_size: dict[str, list[int]] = {}
    per_task: dict[str, list[int]] = {}
    unparsed = 0
    records = []
    for p, raw in zip(problems, outs):
        got = extract_model_answer(raw)
        ok = False
        if got:
            try:
                ok = answers_equal(parse_answer(got),
                                   parse_answer(normalise_reference(p["answer"])),
                                   tol=BENCH_TOL)
            except Exception:
                ok = False
        else:
            unparsed += 1
        per_size.setdefault(p["size"], []).append(int(ok))
        per_task.setdefault(p.get("task") or "?", []).append(int(ok))
        records.append({"size": p["size"], "task": p.get("task"), "id": p.get("id"),
                        "question": p["question"], "reference": p["answer"],
                        "model_answer": got, "correct": bool(ok),
                        "raw": raw})
    return {"per_size": per_size, "per_task": per_task, "unparsed": unparsed,
            "records": records}


def report(res: dict, label: str, problems: list[dict]) -> None:
    per_size = res["per_size"]
    per_task = res["per_task"]
    print("\n" + "=" * 72)
    print(f"  {label}")
    print("=" * 72)
    unasked = [p for p in problems if not p.get("instructed", True)]
    if unasked:
        sizes = sorted({p["size"] for p in unasked})
        print(f"  NOTE: {len(unasked)} of these problems state no task at all -")
        print(f"        the {', '.join(sizes)} questions in LinAlg-Bench are a bare")
        print("        matrix with no instruction. Nothing tells the model what is")
        print("        being asked, so a score on that column measures the data,")
        print("        not the model. Every model scores near zero there.")
    print(f"  {'size':<8}{'n':>6}{'correct':>10}{'accuracy':>12}")
    for size in ("3x3", "4x4", "5x5"):
        v = per_size.get(size)
        if not v:
            continue
        print(f"  {size:<8}{len(v):>6}{sum(v):>10}{100*sum(v)/len(v):>11.1f}%")
    allv = [x for v in per_size.values() for x in v]
    if allv:
        print(f"  {'overall':<8}{len(allv):>6}{sum(allv):>10}{100*sum(allv)/len(allv):>11.1f}%")
    print(f"  no answer parsed: {res['unparsed']}")
    print("\n  by task:")
    for task in sorted(per_task):
        v = per_task[task]
        print(f"    {task:<24}{sum(v):>4}/{len(v):<4} ({100*sum(v)/len(v):5.1f}%)")
    print("\n  published comparison — LinAlg-Bench accuracy by matrix size")
    print(f"  {'model':<22}{'3x3':>8}{'4x4':>8}{'5x5':>8}")
    for name, (a, b, c) in PUBLISHED_LINALG.items():
        print(f"  {name:<22}{a:>7.1f}%{b:>7.1f}%{c:>7.1f}%")
    print("  source: mst-ai/linalg-bench-llm, results/accuracy/")


def measure_perf(tok, model, prompts: list[str], batch_sizes=(1, 4, 8),
                 max_new_tokens: int = 1024) -> dict:
    """Time to first token, decode throughput and peak VRAM."""
    import torch
    out: dict = {}
    try:
        out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception:
        out["peak_vram_gb"] = None
    for bs in batch_sizes:
        chunk = (prompts * bs)[:bs]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        dt = time.time() - t0
        new = gen.shape[1] - enc["input_ids"].shape[1]
        out[f"batch_{bs}"] = {
            "seconds": round(dt, 2),
            "tokens_generated": int(new * bs),
            "tokens_per_second": round(new * bs / dt, 1) if dt > 0 else None,
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark against published results")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--adapter", default="runs/vslm")
    ap.add_argument("--benchmark", default="linalg",
                    choices=["linalg", "internal"], help="which benchmark to run")
    ap.add_argument("--data", default="data/test.jsonl",
                    help="the internal held-out set, when --benchmark internal")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline", action="store_true",
                    help="also score the base model with no adapter, for the delta")
    ap.add_argument("--perf", action="store_true", help="measure speed and memory")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    if args.benchmark == "linalg":
        problems = fetch_linalg()
        if args.limit:
            problems = problems[: args.limit]
        print(f"{len(problems)} LinAlg-Bench problems")
    else:
        rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
        if args.limit:
            rows = rows[: args.limit]
        problems = [{"size": str(r.get("difficulty", "?")), "question": r["question"],
                     "answer": r["truth"], "id": r.get("family", ""),
                     "task": r.get("family", "")} for r in rows]
        print(f"{len(problems)} internal held-out problems")

    adapter = args.adapter if os.path.isdir(args.adapter) else None
    tok, model = load_model(args.base, adapter)
    t0 = time.time()
    outs = generate(tok, model, [p["prompt"] for p in problems],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, batch_size=args.batch_size)
    print(f"generation took {(time.time()-t0)/60:.1f} min")
    res = score(problems, outs)
    report(res, f"fine-tuned model — {args.benchmark}", problems)

    if args.baseline:
        print("\nscoring the base model with no adapter, for the delta ...")
        # the fine-tuned model is freed to make room, but the perf section
        # below still wants one, so it is kept unless perf is not asked for
        if not args.perf:
            del model
        import torch
        torch.cuda.empty_cache()
        tok2, base_model = load_model(args.base, None)
        t0 = time.time()
        base_outs = generate(tok2, base_model, [p["prompt"] for p in problems],
                             max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, batch_size=args.batch_size)
        print(f"generation took {(time.time()-t0)/60:.1f} min")
        bres = score(problems, base_outs)
        report(bres, "base model, no adapter", problems)
        del base_model
        torch.cuda.empty_cache()
        fa = res["per_size"]
        ba = bres["per_size"]
        print("\n  delta (fine-tuned minus base)")
        for size in ("3x3", "4x4", "5x5"):
            if size in fa and size in ba:
                a = 100 * sum(fa[size]) / len(fa[size])
                b = 100 * sum(ba[size]) / len(ba[size])
                print(f"    {size:<6}{b:6.1f}%  ->  {a:6.1f}%   ({a-b:+.1f} points)")
        res["baseline_records"] = bres["records"]

    if args.perf:
        print("\n" + "=" * 72)
        print("  performance")
        print("=" * 72)
        prompts = [p["question"] for p in problems[:8]] or ["What is the determinant of [[2,1],[1,2]]?"]
        perf = measure_perf(tok, model, prompts)
        if perf.get("peak_vram_gb"):
            print(f"  peak VRAM allocated: {perf['peak_vram_gb']} GB")
        for k, v in perf.items():
            if isinstance(v, dict):
                print(f"  batch {k.replace('batch_','')}: {v['tokens_generated']} tokens in "
                      f"{v['seconds']} s = {v['tokens_per_second']} tok/s")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for rec in res["records"]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nwrote per-example results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
