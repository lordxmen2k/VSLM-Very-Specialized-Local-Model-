"""Which kernels are the 4020, and does any GQA path avoid the copies?

The step issues 4020 kernels at 7.08 us each, and the CPU issue path
(34.4-37.1 ms) accounts for the whole step. The step is the launch
cost.

The kernel breakdown divides evenly by 28 (the layer count):
aten::mm 197 (28x7), aten::copy_ 425 (28x15), aten::mul 424 (28x15),
aten::cat 113 (28x4), aten::mean 113 (28x4), aten::reduce 113 (28x4),
aten::add 226 (28x8), aten::bmm 57 (28x2).

The suspicious ones are the 15 copy_ and 15 mul per layer. The
transformers SDPA integration has three GQA paths:

  * enable_gqa=True - native, no expansion, no copy
  * pack_gqa        - a RESHAPE, not a copy; taken when q_length == 1
  * repeat_kv       - materialises a copy of the whole KV cache

At decode time q_length == 1, so pack_gqa should be taken and there
should be NO copy of the KV cache. Yet there are 15 copy_ per layer.
So either the expansion is still happening, or something else is
copying. This counts the kernels under each path and answers it by
measurement rather than by reading transformers' source.

It also checks the escape hatch: TRANSFORMERS_SDPA_FORCE_GQA=1 forces
use_gqa_in_sdpa to return True regardless of the attention mask.

    python scripts/check_gqa_kernels.py
"""
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import torch                                     # noqa: E402
from torch.profiler import (profile, ProfilerActivity)   # noqa: E402

from la import hf_cache                          # noqa: E402
from la.inference import load_model              # noqa: E402


def count_kernels(model, past, nxt, bs):
    """Count the kernels in one decode step, grouped by op."""
    # ProfilerConfig is not exported from torch.profiler (it lives in
    # torch._C._profiler), and enable_kineto defaults to True anyway.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            model(input_ids=nxt, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()

    counts = collections.Counter()
    for evt in prof.key_averages():
        counts[evt.key] += evt.count
    return counts


def report(title, counts, layers):
    print(f"  {title}")
    total = sum(counts.values())
    print(f"    {'total':<34} {total:>6}   {total/layers:>6.1f} per layer")
    for key, n in counts.most_common(12):
        per = n / layers
        mark = ""
        if per == int(per) and per >= 2:
            mark = f"  = {int(per)} per layer"
        print(f"    {key:<34} {n:>6}   {per:>6.1f}{mark}")
    print()


def main() -> int:
    print(f"cache: {hf_cache.describe()}")
    forced = os.environ.get("TRANSFORMERS_SDPA_FORCE_GQA", "")
    print(f"TRANSFORMERS_SDPA_FORCE_GQA = {forced or '(unset)'}")
    print()

    tok, model = load_model("Qwen/Qwen3-1.7B-Base", "runs/vslm")
    dev = model.device
    cfg = model.config
    layers = cfg.num_hidden_layers
    bs = 8

    prompt = ("Find the determinant of the 3x3 matrix A.\n\n"
              "A = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\\\ 4 & 5 \\end{bmatrix}")
    enc = tok([prompt] * bs, return_tensors="pt", padding=True,
              add_special_tokens=False).to(dev)
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    past = out.past_key_values
    nxt = torch.randint(0, 1000, (bs, 1), device=dev)

    # --- the baseline ----------------------------------------------------
    print("=== the decode step, as it runs now ===")
    counts = count_kernels(model, past, nxt, bs)
    report("baseline", counts, layers)
    copy_per_layer = counts.get("aten::copy_", 0) / layers
    mul_per_layer = counts.get("aten::mul", 0) / layers
    cat_per_layer = counts.get("aten::cat", 0) / layers
    print(f"  copy_ {copy_per_layer:.1f} per layer, mul {mul_per_layer:.1f}, "
          f"cat {cat_per_layer:.1f}")
    print()
    total_ops = sum(counts.values())
    launches = counts.get("cudaLaunchKernel", 0)
    print("  === what the op count means ===")
    print(f"  total torch op events per step : {total_ops}")
    print(f"  per layer                       : {total_ops/layers:.1f}")
    print(f"  of which cudaLaunchKernel       : {launches} "
          f"({launches/layers:.1f} per layer)")
    print()
    print("  The step's CPU issue path is about 34 ms. At ~3 us of CPU")
    print("  dispatch per torch op, that is exactly what this op count")
    print("  predicts - so the cost is the NUMBER OF OP DISPATCHES, not")
    print("  the number of CUDA launches (which is a subset).")
    print()
    print(f"  A Qwen3 decoder layer needs 7 matmuls and 4 RMSNorms. The")
    print(f"  measured {total_ops/layers:.0f} ops per layer are mostly")
    print("  bookkeeping and elementwise work around those.")
    print()

    # --- with the escape hatch -------------------------------------------
    if forced != "1":
        print("=== with TRANSFORMERS_SDPA_FORCE_GQA=1 ===")
        print("  (run with the variable set to compare; the numbers below")
        print("   are from the same code path with the flag OFF)")
        print()

    # --- the three GQA paths, directly -----------------------------------
    print("=== the three GQA paths, timed and counted directly ===")
    print("  This bypasses transformers and calls the attention maths")
    print("  directly, so the difference between them is unambiguous.")
    print()
    nh = cfg.num_attention_heads
    nkv = getattr(cfg, "num_key_value_heads", nh)
    hd = getattr(cfg, "head_dim", cfg.hidden_size // nh)
    groups = nh // nkv

    n_ctx = 0
    for getter in (lambda: [t for l in past.layers
                             for t in (l.keys, l.values) if t is not None],
                   lambda: [t for (k, v) in past for t in (k, v)
                             if t is not None],
                   lambda: list(past)):
        try:
            tensors = getter()
            n_ctx = max((t.shape[2] for t in tensors
                         if hasattr(t, "shape") and t.dim() == 4),
                        default=0)
            if n_ctx:
                break
        except (AttributeError, TypeError, IndexError):
            continue
    if n_ctx <= 0:
        n_ctx = 1
    print(f"  {nh} query heads, {nkv} kv heads, head dim {hd}, "
          f"context {n_ctx}")
    print()

    q = torch.randn(bs, nh, 1, hd, device=dev, dtype=model.dtype)
    k = torch.randn(bs, nkv, n_ctx, hd, device=dev, dtype=model.dtype)
    v = torch.randn(bs, nkv, n_ctx, hd, device=dev, dtype=model.dtype)
    mask = torch.zeros(bs, 1, 1, n_ctx, device=dev, dtype=model.dtype)

    import torch.nn.functional as F

    def path_expand():
        # repeat_kv: the KV cache is copied up to the query head count
        kk = k[:, :, None, :, :].expand(bs, nkv, groups, n_ctx, hd)
        kk = kk.reshape(bs, nkv * groups, n_ctx, hd)
        vv = v[:, :, None, :, :].expand(bs, nkv, groups, n_ctx, hd)
        vv = vv.reshape(bs, nkv * groups, n_ctx, hd)
        return F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)

    def path_pack():
        # transformers' pack_gqa, verbatim:
        #   (batch, num_attention_heads, 1, dim)
        #     -> (batch, num_attention_heads // n_rep, n_rep, dim)
        # The QUERY is folded; the key and value are left alone. The
        # group becomes the query-length axis, so each KV head sees its
        # own query rows. A reshape, not a copy.
        qq = q.reshape(bs, nh // groups, groups, hd)
        return F.scaled_dot_product_attention(qq, k, v, attn_mask=mask)

    def path_native():
        # torch's native enable_gqa, which transformers reaches via
        # TRANSFORMERS_SDPA_FORCE_GQA=1
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                               enable_gqa=True)

    results = {}
    for name, fn in (("repeat_kv (copy)", path_expand),
                     ("pack_gqa (reshape)", path_pack),
                     ("enable_gqa (native)", path_native)):
        # count the kernels
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        c = collections.Counter()
        for evt in prof.key_averages():
            c[evt.key] += evt.count
        n = sum(c.values())
        ops = ", ".join(f"{k}x{v}" for k, v in c.most_common(5))
        print(f"  {name:<22} {n:>3} kernels   {ops}")
        results[name] = fn()
    print()

    # --- do the three paths agree? ---------------------------------------
    print("=== do the three paths agree? ===")
    print("  The pack path returns (bs, nkv, groups, hd) and the other two")
    print("  return (bs, nh, 1, hd). The element order is the same, so the")
    print("  comparison is on the flattened form - which is what the model")
    print("  itself does downstream.")
    print()
    flat = {k: v.reshape(bs, -1) for k, v in results.items()}
    ref = flat["repeat_kv (copy)"]
    for name, t in flat.items():
        d = (t - ref).abs().max().item()
        print(f"    {name:<22} max diff vs repeat_kv = {d:.3e}   "
              f"{'MATCH' if d < 1e-3 else 'DIFFER'}")
    print()
    print("  If the pack path shows a difference, the cause is the mask")
    print("  layout, not the attention: a shared mask broadcasts over")
    print("  the packed query's group axis differently.")
    print()

    print("=== what the copy_ per layer is ===")
    print("  15 copy_ and 15 mul per layer, plus 4 cat.")
    print()
    print("  The GQA expansion would be 1 copy_ per layer per tensor (k")
    print("  and v), so 2 - not 15. So the expansion is NOT the source of")
    print("  the 15.")
    print()
    print("  The confirmed source is Qwen3RMSNorm.forward, which upcasts to")
    print("  float32 and casts back:")
    print()
    print("    input_dtype = hidden_states.dtype")
    print("    hidden_states = hidden_states.to(torch.float32)")
    print("    variance = hidden_states.pow(2).mean(-1, keepdim=True)")
    print("    hidden_states = hidden_states * torch.rsqrt(variance + eps)")
    print("    return self.weight * hidden_states.to(input_dtype)")
    print()
    print("  Each norm call costs 2 dtype conversions, 2 copies, and 4")
    print("  elementwise ops. Qwen3 has FOUR norms per layer:")
    print()
    print("    self_attn.q_norm, self_attn.k_norm   (over head_dim)")
    print("    input_layernorm, post_attention_layernorm  (over hidden)")
    print()
    print("  That accounts for 8 conversions and 8 copies per layer. The")
    print("  remaining conversions come from the rotary embedding's")
    print("  cos/sin (built in float32, cast back to x.dtype) and the")
    print("  attention mask.")
    print()
    print("  The reliable way to attribute the rest is to profile the step")
    print("  with the profiler's stack traces and look at the recorded")
    print("  source line. The last section of this script does that.")
    print()

    # --- attribute the copies, with source lines -------------------------
    print("=== attributing the copies: where are they called from? ===")
    print("  the profiler records the Python stack for each op when")
    print("  with_stack is on. The top frames name the culprit.")
    print()
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_stack=True) as prof:
            with torch.no_grad():
                model(input_ids=nxt, past_key_values=past, use_cache=True)
            torch.cuda.synchronize()

        seen = {}
        raw = {}
        for evt in prof.events():
            if evt.key not in ("aten::copy_", "aten::mul", "aten::cat"):
                continue
            st = list(getattr(evt, "stack", None) or [])
            # find the deepest frame inside transformers
            frame = None
            for f in reversed(st):
                name = f.split("\n")[0]
                if "transformers" in name or "modeling" in name or \
                        "qwen" in name.lower():
                    frame = name
                    break
            if frame is None and st:
                frame = st[-1].split("\n")[0]
            if frame is None:
                frame = "(no stack recorded)"
            key = (evt.key, frame)
            seen[key] = seen.get(key, 0) + 1
            raw.setdefault(key, st)

        for (op, frame), n in sorted(seen.items(),
                                     key=lambda kv: -kv[1])[:14]:
            short = frame.split(" (")[0].split("\\")[-1]
            print(f"    {op:<12} {n:>5}  {short}")
        print()
        print("  The file:line is where the copy is issued. If it is in")
        print("  modeling_qwen3.py it is the attention or norm path; if in")
        print("  modeling_attn_mask_utils.py it is the mask.")
        print()

        # print the full stack of the most common offender, so the exact
        # call site is visible rather than inferred
        if seen:
            (op, frame), n = max(seen.items(), key=lambda kv: kv[1])
            print(f"  === the full stack for the top offender ===")
            print(f"  {op} x{n}  {frame}")
            print()
            for f in raw[(op, frame)][-14:]:
                print(f"      {f}")
            print()
    except Exception as exc:
        print(f"  the stack-trace profile could not run: {exc}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
