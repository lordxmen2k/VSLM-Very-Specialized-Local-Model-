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


class ProblemDataset(Dataset):
    """Prompt tokens are masked; only the completion is supervised."""

    def __init__(self, rows: list[dict], tokenizer, max_len: int = 2048):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        prompt = row["prompt"] if "prompt" in row else \
            f"### Problem\n{row['question']}\n\n### Scratchpad\n"
        completion = row["completion"] if "completion" in row else \
            f"{row['scratchpad']}\n### Answer\n{row['answer']}\n### Plot\n" \
            f"{json.dumps(row.get('spec') or {}, separators=(',', ':'))}\n"
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = self.tok(completion, add_special_tokens=False)["input_ids"]
        # leave room for the sequence end token
        ids = (p_ids + c_ids)[: self.max_len]
        labels = ([IGNORE] * len(p_ids) + c_ids)[: self.max_len]
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="QLoRA fine-tune Qwen3-1.7B")
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--out", default="runs/vslm")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
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

    tok, model = build_model(args.base, None, args.rank, args.alpha,
                              load_4bit=want_4bit)
    ds = ProblemDataset(rows, tok, args.max_seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
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

    os.makedirs(args.out, exist_ok=True)
    model.train()
    step = 0
    micro = 0
    t0 = time.time()
    run_loss = 0.0
    done = False
    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        for batch in dl:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            run_loss += out.loss.item()
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
            if step % args.log_every == 0:
                avg = run_loss / (args.log_every * args.grad_accum)
                el = time.time() - t0
                print(f"  step {step:>6,}/{total_steps:,}  loss {avg:.4f}  "
                      f"lr {lr:.2e}  {el/60:.1f} min  "
                      f"eta {el/step*(total_steps-step)/60:.1f} min", flush=True)
                run_loss = 0.0
            if step >= total_steps:
                done = True
                break

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)
    print(f"saved adapter to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
