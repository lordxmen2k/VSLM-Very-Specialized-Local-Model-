"""Why is the benchmark taking 35 minutes per batch of 8?

The run is at ~35 min for 2048 tokens at batch 8. At the 34 ms/step that
was measured on the bf16 base, 2048 steps is 70 seconds. So the benchmark
model is doing something ~30x slower than the model that was profiled, or
it is generating far more tokens than expected.

This script answers that by running a handful of benchmark problems and
printing, for each one:

  * how many tokens it actually generated
  * whether the stop token was emitted
  * the last 300 characters of the completion
  * how long it took, and the tokens per second

It also compares the adapter model against the plain base on the SAME
prompts, so the delta is attributable.

    python scripts/check_bench_run.py --n 4
    python scripts/check_bench_run.py --n 4 --max-new-tokens 256

The cap defaults to 512 so the script cannot run for hours. If a problem
hits the cap, it did not stop - and the printout says so.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from la import hf_cache                          # noqa: E402
from la.benchprompt import benchmark_prompt_pair, render_chat  # noqa: E402
from la.inference import load_model, stop_token_ids            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default="runs/vslm")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-base", action="store_true",
                    help="do not also run the plain base model")
    args = ap.parse_args()

    print(f"cache: {hf_cache.describe()}")
    print()

    # --- the prompts ------------------------------------------------------
    from la.bench import fetch_linalg
    ps = fetch_linalg("bench_cache")
    print(f"  {len(ps)} benchmark problems", flush=True)
    pairs = []
    for p in ps[: args.n]:
        pairs.append((p.get("system", ""), p.get("user", "")))
    print()

    def run(label, tok, model, max_new_tokens):
        eos = stop_token_ids(tok, model)
        print(f"=== {label} ===")
        print(f"  eos_token_id      = {eos}")
        print(f"  tok.eos_token_id  = {tok.eos_token_id}")
        print(f"  tok.eos_token     = {tok.eos_token!r}")
        im = tok.convert_tokens_to_ids("<|im_end|>")
        print(f"  <|im_end|> id     = {im}")
        try:
            print(f"  generation_config = "
                  f"{model.generation_config.eos_token_id}")
        except Exception as exc:
            print(f"  generation_config = (could not read: {exc})")
        print(f"  dtype             = {model.dtype}")
        print()

        prompts = [render_chat(tok, s, u) for s, u in pairs]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        t = time.time()
        import torch
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 eos_token_id=eos)
        dt = time.time() - t
        plen = enc["input_ids"].shape[1]
        print(f"  {dt:.1f}s for {len(prompts)} problems "
              f"({dt/max(len(prompts),1):.1f}s each)")
        print()
        stopped = 0
        for j in range(len(prompts)):
            new = gen[j][plen:]
            n = new.shape[0]
            hit = n < max_new_tokens
            stopped += 1 if hit else 0
            text = tok.decode(new, skip_special_tokens=True)
            print(f"  problem {j}: {n} tokens, "
                  f"{'STOPPED' if hit else 'HIT THE CAP'}")
            print(f"    tail: {text[-300:]!r}")
            print()
        print(f"  stopped: {stopped}/{len(prompts)}")
        if stopped < len(prompts):
            print(f"  the ones that hit the cap did NOT stop.")
        print()
        return dt, stopped, len(prompts)

    # --- the adapter ------------------------------------------------------
    tok, model = load_model(args.base, args.adapter)
    run(f"the adapter ({args.adapter})", tok, model, args.max_new_tokens)
    del model
    import torch
    torch.cuda.empty_cache()

    # --- the plain base ---------------------------------------------------
    if not args.skip_base:
        tok2, base = load_model(args.base, None)
        run("the plain base, no adapter", tok2, base, args.max_new_tokens)

    print("=" * 62)
    print("If the adapter's completions end with the answer and a stop,")
    print("the model stops. If they hit the cap, the model is not")
    print("emitting the stop token, and every problem costs the full")
    print("cap - which is what 35 minutes a batch means.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
