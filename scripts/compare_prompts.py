"""Why does a benchmark batch take minutes?

The first batch of the LinAlg-Bench run did not finish in six minutes. The
model is healthy - the KV cache is on, the dtype is bf16, the attention
implementation is sdpa, the card is idle - and on a prompt it recognises the
model stops by itself after about two hundred tokens.

That leaves the prompt as the suspect. The corpus was trained on the question
alone. The benchmark sends the question with the benchmark's own system prompt
in front of it, and that prefix is the one thing the model has never seen.

This script runs the same eight problems twice: once with the benchmark's
prompt, once with the corpus's. No download, no network, and it prints as it
goes.

    python scripts/compare_prompts.py

It needs the benchmark CSVs in ./bench_cache. If they are not there, get them
with:

    python -m la.bench --limit 1 --adapter '' 2>&1 | head -1
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

CACHE = "bench_cache"

from la.bench import fetch_linalg                      # noqa: E402
from la.inference import load_model                    # noqa: E402


def run(tok, model, prompts, label, max_new_tokens=2048):
    enc = tok(prompts, return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)
    print(f"{label}  prompt shape {tuple(enc['input_ids'].shape)}", flush=True)
    t = time.time()
    with __import__("torch").no_grad():
        g = model.generate(**enc, max_new_tokens=max_new_tokens,
                           do_sample=False, pad_token_id=tok.pad_token_id)
    dt = time.time() - t
    n = [g[j][enc["input_ids"].shape[1]:].shape[0]
         for j in range(len(prompts))]
    print(f"   {dt:6.1f}s   new tokens {n}", flush=True)
    return dt, n


def main() -> int:
    print("loading the model ...", flush=True)
    tok, model = load_model("Qwen/Qwen3-1.7B-Base", "runs/vslm")
    print(f"  use_cache = {model.config.use_cache}", flush=True)
    print(f"  dtype     = {model.dtype}", flush=True)

    print("\nloading the benchmark problems ...", flush=True)
    missing = [f"bench_cache/linalg_bench_{s}.csv" for s in ("3x3", "4x4", "5x5")
               if not os.path.exists(f"bench_cache/linalg_bench_{s}.csv")]
    if missing:
        print(f"  MISSING: {', '.join(missing)}", flush=True)
        print("  Fetch them once with:", flush=True)
        print("    python -c \"from la.bench import fetch_linalg; fetch_linalg('bench_cache')\"", flush=True)
        print("  (it prints 'downloading ...' per file, so it is not silent)", flush=True)
        return 1
    ps = fetch_linalg(CACHE)
    print(f"  {len(ps)} problems", flush=True)

    bench = [p["prompt"] for p in ps[:8]]

    # The corpus's own prompt: the question with the benchmark's system
    # prompt taken off the front. The released question already carries the
    # preamble, the task and the closing instruction; only the system prompt
    # is the benchmark's addition, and it is the one thing the model has
    # never seen.
    corpus = []
    for p in ps[:8]:
        text = p["prompt"]
        marker = "\n\nSolve this linear algebra problem."
        i = text.find(marker)
        corpus.append(text[i:] if i > 0 else text)

    print()
    print("  ---- a corpus-style prompt (system prompt removed) ----")
    print("  " + corpus[0][:400].replace("\n", "\n  "))
    print()
    print("  ---- the benchmark prompt (system prompt included) ----")
    print("  " + bench[0][:400].replace("\n", "\n  "))

    print()
    dt_c, n_c = run(tok, model, corpus, "corpus-style prompt")
    print()
    dt_b, n_b = run(tok, model, bench, "benchmark prompt   ")
    print()

    print("=" * 60)
    print(f"corpus-style : {dt_c:6.1f}s   tokens {n_c}")
    print(f"benchmark    : {dt_b:6.1f}s   tokens {n_b}")
    print()
    if dt_b > 2 * dt_c:
        print("=> The benchmark prompt is the problem. The model rambles under a")
        print("   prefix it was never trained on and never emits EOS, so every")
        print("   problem costs the full max_new_tokens.")
        print("   FIX: train on the benchmark's prompt, and use the instruct")
        print("   checkpoint, whose weights are trained to emit <|im_end|>.")
    else:
        print("=> The prompts behave the same, so the prompt is not the cause.")
        print("   Look at the decode rate instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
