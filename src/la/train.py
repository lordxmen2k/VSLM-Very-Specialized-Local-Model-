"""QLoRA fine-tuning for Qwen3-1.7B-Base on a 12 GB card.

This is a plain PyTorch loop rather than a Trainer class, for two reasons: the
prompt tokens must be masked out of the loss exactly, and the loop has to
survive the churn in the training-library APIs without changing.

    python -m la.train --data data/train.jsonl --out runs/vslm

The default settings are chosen for a 12 GB card: 4-bit NF4 weights, rank-32
LoRA, gradient checkpointing, a sequence length of 2048 and 16 steps of
gradient accumulation. A 1.7B model at that length fits in about 8 GB of
activations and optimiser state, so the card has room for a 4K context if you
raise --max-seq-len.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import random
import sys
import time

import torch
from torch.utils.data import Dataset, DataLoader

IGNORE = -100


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def tokenise_rows(rows: list[dict], tokenizer, max_len: int | None = 2048):
    """Tokenise every row into (input_ids, labels).

    Prompt tokens are masked, so only the completion is supervised. A
    max_len of None leaves the sequence untruncated, which is how the speed
    test measures what the corpus actually needs.
    """
    items: list[tuple[list[int], list[int]]] = []
    for row in rows:
        prompt = row["prompt"] if "prompt" in row else \
            f"### Problem\n{row['question']}\n\n### Scratchpad\n"
        completion = row["completion"] if "completion" in row else \
            f"{row['scratchpad']}\n### Answer\n{row['answer']}\n### Plot\n" \
            f"{json.dumps(row.get('spec') or {}, separators=(',', ':'))}\n"
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if max_len is not None:
            ids = (p_ids + c_ids)[:max_len]
            labels = ([IGNORE] * len(p_ids) + c_ids)[:max_len]
        else:
            ids = p_ids + c_ids
            labels = [IGNORE] * len(p_ids) + c_ids
        items.append((ids, labels))
    return items


class ProblemDataset(Dataset):
    """Prompt tokens are masked; only the completion is supervised.

    The corpus is tokenised once, here, rather than on every access. A
    training run reads every row once per epoch, and a loader that calls
    the tokenizer in the hot loop shows up as a GPU sitting idle between
    batches - which is indistinguishable from a slow GPU until you measure
    the two separately.
    """

    def __init__(self, rows: list[dict], tokenizer, max_len: int = 2048):
        self.max_len = max_len
        self.items = tokenise_rows(rows, tokenizer, max_len)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        ids, labels = self.items[i]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id: int):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * k)
        labels.append(b["labels"] + [IGNORE] * k)
        attn.append([1] * len(b["input_ids"]) + [0] * k)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def bitsandbytes_works() -> tuple[bool, str]:
    """Probe whether bitsandbytes can actually run on this machine.

    Importing it is not enough. The package imports cleanly and then fails
    later, deep inside model loading, when it cannot find a native binary
    matching the installed CUDA version - which is what happens on a torch
    built for a CUDA release that bitsandbytes has no wheel for.

    Returns (usable, reason).
    """
    try:
        import bitsandbytes as bnb
    except Exception as exc:
        return False, f"bitsandbytes will not import: {exc}"
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    try:
        # a real quantisation on the real device, which is the thing that
        # actually fails when the native binary is missing or mismatched
        t = torch.zeros(16, 16, device="cuda", dtype=torch.bfloat16)
        bnb.functional.quantize_4bit(t, quant_type="nf4")
        return True, ""
    except Exception as exc:
        first = str(exc).strip().split("\n")[0][:120]
        return False, first


def build_model(base: str, adapter_out: str | None, rank: int, alpha: int,
                load_4bit: bool = True, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    qcfg = None
    if load_4bit:
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=qcfg, dtype=dtype, device_map={"": 0},
        trust_remote_code=True,
    )
    if load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # with a frozen bf16 base, gradient checkpointing needs the embedding
        # output to require grad or the backward pass stops at the first block
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lcfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return tok, model


def cosine_lr(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def speed_one(args, rows, tok, model, optim, pad_id, bs: int, ga: int) -> dict:
    """Run one micro-batch shape and report what it cost.

    Called in a process of its own. The caching allocator does not give back
    memory it has cached, and it cannot always reuse a block for a request
    of a different size, so a process that has already run a wider
    configuration measures the next one through its own fragmentation.
    Measuring several shapes in one process therefore measures the sum of
    them, not each one.
    """
    ds = ProblemDataset(rows, tok, args.max_seq_len)
    dl = DataLoader(ds, batch_size=bs, shuffle=True,
                    collate_fn=lambda b: collate(b, pad_id))
    optim.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    warm, measure = 2, 5
    micro = step = 0
    cur_tok = 0
    tok_n = ex_n = 0
    t0 = time.time()
    try:
        for batch in dl:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            cur_tok += int(batch["attention_mask"].sum().item())
            out = model(**batch)
            (out.loss / ga).backward()
            micro += 1
            if micro % ga:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step == warm:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
            elif step > warm:
                tok_n += cur_tok
                ex_n += bs
            cur_tok = 0
            if step >= warm + measure:
                break
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"bs": bs, "ga": ga, "oom": True}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    el = time.time() - t0
    steps = max(1, step - warm)
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / 1e9
        resv = torch.cuda.max_memory_reserved() / 1e9
    else:
        alloc = resv = 0.0
    return {"bs": bs, "ga": ga, "oom": False,
            "s_step": el / steps, "tok_s": tok_n / max(el, 1e-9),
            "ex_s": ex_n / max(el, 1e-9), "alloc": alloc, "resv": resv}


def speed_test(args, rows, tok, pad_id) -> int:
    """Measure the cost of several micro-batch shapes, each in its own process.

    Every shape is the same effective batch of 16, so every shape sees the
    same number of examples per optimiser step. If the time per step barely
    moves, the cost is per token and the levers are fewer tokens or fewer
    examples. If it moves a lot, the micro-batch shape was the problem.

    The shapes are measured in separate processes because the caching
    allocator carries its fragmentation from one to the next. Measuring them
    in one process gives a different answer each time it is run, which is
    the giveaway that the instrument is measuring the wrong thing.
    """
    configs = SPEED_CONFIGS
    print("\n" + "=" * 72)
    print("  speed test")
    print("=" * 72)

    uncapped = tokenise_rows(rows, tok, None)
    lengths = sorted(len(ids) for ids, _ in uncapped)
    n = len(lengths)
    cap = args.max_seq_len
    cut = sum(1 for L in lengths if L > cap)
    print(f"  corpus: {n} examples, {sum(lengths):,} tokens, "
          f"mean {sum(lengths)/n:.0f} tokens, median {lengths[n//2]}, "
          f"p95 {lengths[int(n*0.95)]}, p99 {lengths[int(n*0.99)]}, "
          f"max {lengths[-1]}")
    print(f"  sequence cap: {cap} tokens")
    if cut:
        print(f"  TRUNCATED: {cut}/{n} examples ({100*cut/n:.1f}%) are cut at "
              f"the cap.")
        print("  A truncated example loses the end of its completion, which is")
        print("  where the answer and the plot spec are. Raise --max-seq-len")
        print("  until this is zero, and check the memory column below.")
    else:
        print("  truncation: none")
    card = torch.cuda.get_device_properties(0).total_memory / 1e9 \
        if torch.cuda.is_available() else 0.0
    print()
    print(f"  {'batch':>6}{'accum':>7}{'eff':>5}{'s/step':>9}"
          f"{'tok/s':>10}{'ex/s':>8}{'alloc GB':>10}{'resv GB':>9}{'frag':>7}")

    for bs, ga in configs:
        cmd = [sys.executable, "-m", "la.train",
               "--speed-config", str(bs), str(ga),
               "--base", args.base, "--data", args.data, "--out", args.out,
               "--max-seq-len", str(args.max_seq_len),
               "--rank", str(args.rank), "--alpha", str(args.alpha),
               "--precision", args.precision]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=1800)
        except subprocess.TimeoutExpired:
            print(f"  {bs:>6}{ga:>7}{bs*ga:>5}   timed out")
            continue
        row = None
        for line in proc.stdout.splitlines():
            if line.startswith(SPEED_ROW_PREFIX):
                row = line
        parsed = parse_speed_row(row)
        if parsed is None:
            err = (proc.stderr or proc.stdout).strip().splitlines()
            detail = err[-1][:60] if err else "no output"
            print(f"  {bs:>6}{ga:>7}{bs*ga:>5}   failed: {detail}")
            continue
        if parsed.get("oom"):
            print(f"  {bs:>6}{ga:>7}{bs*ga:>5}   out of memory")
            continue
        over = "   OVER THE CARD" if card and parsed["resv"] > card else ""
        print(f"  {parsed['bs']:>6}{parsed['ga']:>7}{parsed['eff']:>5}"
              f"{parsed['s_step']:>9.2f}{parsed['tok_s']:>10,.0f}"
              f"{parsed['ex_s']:>8.1f}{parsed['alloc']:>10.2f}"
              f"{parsed['resv']:>9.2f}{parsed['resv']-parsed['alloc']:>7.2f}"
              + over)

    print()
    if card:
        print(f"  the card has {card:.2f} GB. A configuration whose reserved")
        print("  memory exceeds it spills into host memory over PCIe, and on")
        print("  Windows that is several times slower than staying on the card.")
        print("  Take the fastest row that stays under.")
    print()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="QLoRA fine-tune Qwen3-1.7B")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--out", default="runs/vslm")
    ap.add_argument("--epochs", type=float, default=1.0,
                    help="passes over the corpus. 1 is the default because the "
                         "corpus is large enough to fill it; 2 is often a "
                         "little better and costs twice as long")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=768,
                    help="the corpus runs to a median of 344 and a mean of "
                         "358 tokens, with a tail past 512. 768 leaves the "
                         "tail intact; make speed-test report how many "
                         "examples the cap is cutting")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="examples per forward pass. 2 with grad-accum 8 "
                         "keeps the effective batch at 16, and measured "
                         "fastest of the five shapes tried - 1,633 tok/s "
                         "against 814 at batch 1 and 231 at batch 4, which "
                         "spills past the card")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="dataloader workers. The corpus is tokenised up "
                         "front, so 0 is usually enough")
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="gradient accumulation. batch-size x grad-accum is "
                         "the effective batch, which is 16 either way")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop after this many optimiser steps (smoke test)")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only this many training rows")
    ap.add_argument("--precision", choices=["auto", "4bit", "bf16"], default="auto",
                    help="4bit needs bitsandbytes and saves about 2.5 GB; "
                         "bf16 trains a LoRA on a bf16 base and needs about 7 GB "
                         "in total, so it fits a 12 GB card on its own")
    ap.add_argument("--no-4bit", action="store_true",
                    help="deprecated alias for --precision bf16")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--speed-test", action="store_true",
                    help="run the same data through five batch shapes and "
                         "report the cost of each, then exit")
    ap.add_argument("--speed-config", type=int, nargs=2, metavar=("BATCH", "ACCUM"),
                    help="internal: measure one batch shape in this process "
                         "and print one machine-readable row")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        print("no CUDA device found. This script needs a GPU.", file=sys.stderr)
        return 2
    print(f"device: {torch.cuda.get_device_name(0)}")

    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows):,} training rows from {args.data}")

    # ---- choose the precision, and prove the choice works ------------------
    want_4bit = args.precision == "4bit" or (args.precision == "auto" and not args.no_4bit)
    if args.no_4bit and args.precision == "auto":
        want_4bit = False
    if want_4bit:
        ok, reason = bitsandbytes_works()
        if not ok:
            print(f"4-bit quantisation is unavailable: {reason}")
            if args.precision == "4bit":
                print("--precision 4bit was requested explicitly, so stopping. "
                      "See the README section on fixing bitsandbytes, or run "
                      "with --precision bf16.", file=sys.stderr)
                return 2
            print("falling back to a LoRA on a bf16 base. It fits in about 7 GB "
                  "on a 12 GB card, so the run proceeds; only the memory "
                  "headroom is smaller.")
            want_4bit = False
        else:
            print("4-bit quantisation available (NF4, double quant)")
    if not want_4bit:
        print("training a LoRA on a bf16 base: about 7 GB of VRAM in total")

    # print the resolved configuration. The defaults are not on the command
    # line, so `make -n train` looks identical whether the defaults are the
    # old ones or the new ones - the only place the difference is visible
    # is here.
    print("configuration:")
    for k, v in (("base", args.base),
                 ("data", args.data),
                 ("out", args.out),
                 ("epochs", args.epochs),
                 ("batch size", args.batch_size),
                 ("grad accum", args.grad_accum),
                 ("effective batch", args.batch_size * args.grad_accum),
                 ("max seq len", args.max_seq_len),
                 ("learning rate", args.lr),
                 ("lora rank / alpha", f"{args.rank} / {args.alpha}"),
                 ("precision", "4bit NF4" if want_4bit else "bf16"),
                 ("dataloader workers", args.num_workers)):
        print(f"  {k:<18} {v}")

    tok, model = build_model(args.base, None, args.rank, args.alpha,
                              load_4bit=want_4bit)
    ds = ProblemDataset(rows, tok, args.max_seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))

    steps_per_epoch = math.ceil(len(dl) / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = int(total_steps * args.warmup_ratio)
    print(f"{steps_per_epoch:,} optimiser steps per epoch, {total_steps:,} total, "
          f"{warmup} warmup")

    optim = None
    opt_name = "adamw"
    if want_4bit:
        try:
            import bitsandbytes as bnb
            optim = bnb.optim.PagedAdamW8bit(model.parameters(), lr=args.lr,
                                             betas=(0.9, 0.95), weight_decay=0.01)
            opt_name = "paged_adamw_8bit"
        except Exception as exc:
            print(f"paged 8-bit AdamW unavailable ({exc}); using torch AdamW")
    if optim is None:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.01)
    print(f"optimiser: {opt_name}")

    if args.speed_config:
        # one configuration, in this process, so the allocator starts clean
        row = speed_one(args, rows, tok, model, optim, tok.pad_token_id,
                         args.speed_config[0], args.speed_config[1])
        if row["oom"]:
            print(format_oom())
        else:
            print(format_row(row["bs"], row["ga"], row["s_step"], row["tok_s"],
                             row["ex_s"], row["alloc"], row["resv"]))
        return 0

    if args.speed_test:
        return speed_test(args, rows[: args.limit or 2000], tok, tok.pad_token_id)

    os.makedirs(args.out, exist_ok=True)
    model.train()
    step = 0
    micro = 0
    t0 = time.time()
    last_step_t = t0
    run_loss = 0.0
    win_micro = 0
    done = False
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    data_s = 0.0
    tok_count = 0
    win_tok = 0
    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        it = iter(dl)
        while True:
            t_fetch = time.time()
            try:
                batch = next(it)
            except StopIteration:
                break
            data_s += time.time() - t_fetch
            batch = {k: v.to(model.device) for k, v in batch.items()}
            win_tok += int(batch["attention_mask"].sum().item())
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            run_loss += out.loss.item()
            win_micro += 1
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = cosine_lr(step, total_steps, warmup, args.lr)
            for g in optim.param_groups:
                g["lr"] = lr
            optim.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            now = time.time()
            step_s = now - last_step_t
            last_step_t = now
            # log the first step, the last step, and every log_every between
            # them, so a short smoke test still shows a curve
            if (step % args.log_every == 0 or step == 1 or step == total_steps):
                # average over the micro-batches actually accumulated since
                # the last log, not over a full logging window - at step 1
                # there has only been one
                avg = run_loss / max(win_micro, 1)
                el = now - t0
                line = (f"  step {step:>6,}/{total_steps:,}  loss {avg:.4f}  "
                        f"lr {lr:.2e}  {step_s:.2f} s/step  "
                        f"{el/60:.1f} min  "
                        f"eta {el/step*(total_steps-step)/60:.1f} min")
                if torch.cuda.is_available():
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    res = torch.cuda.max_memory_reserved() / 1e9
                    line += f"  vram {peak:.1f}/{res:.1f} GB"
                line += f"  data {data_s:.1f}s"
                if step_s > 0:
                    line += f"  {win_tok/step_s:,.0f} tok/s"
                data_s = 0.0
                win_tok = 0
                print(line, flush=True)
                run_loss = 0.0
                win_micro = 0
            if step >= total_steps:
                done = True
                break
        if done:
            break

    if torch.cuda.is_available():
        print(f"peak vram: {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
              f"allocated, {torch.cuda.max_memory_reserved()/1e9:.2f} GB reserved")
    total_s = time.time() - t0
    print(f"{step:,} optimiser steps in {total_s/60:.1f} min "
          f"({total_s/max(step,1):.2f} s/step)")

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)
    print(f"saved adapter to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
