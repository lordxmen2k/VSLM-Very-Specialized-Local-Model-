"""Shared inference helpers: load, generate, and parse a completion.

The model's output is three labelled blocks:

    ### Scratchpad
    ...working...

    ### Answer
    ...LaTeX...

    ### Plot
    {...json...}

Parsing is deliberately tolerant. A model that produces a good answer and a
broken plot should still be credited with the answer, so a failure in one
block does not discard the others.
"""
from __future__ import annotations

import json
import os
import re
import time
import threading

from typing import Any

# The Hugging Face cache has to be pointed at a drive with room before
# huggingface_hub is imported, because it resolves the path once, at import
# time. This import must come before transformers.
from . import hf_cache  # noqa: F401

import torch

from .progress import HEARTBEAT_SECONDS, progress_line

from .parse import extract_json, parse_completion, summarise_answer


def load_model(base: str, adapter: str | None = None, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # The tokenizer comes from the BASE, never from the adapter.
    #
    # It used to be `adapter or base`, and that turned a crashed training
    # run into a confusing error. The adapter directory is created before
    # the training loop and the tokenizer is only written into it at the
    # very end, so a run that crashed left an empty directory - and every
    # later command died with
    #
    #     ValueError: Couldn't instantiate the backend tokenizer
    #
    # which reads like a missing dependency rather than a missing training
    # run. The tokenizer is the base model's; the adapter holds weights.
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=dtype, device_map={"": 0}, trust_remote_code=True)
    if adapter and os.path.isdir(adapter):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


from .benchprompt import DEFAULT_THINKING, model_prompt as build_prompt, render_chat


def stop_token_ids(tok, model=None) -> list[int] | None:
    """The tokens that mean the model has finished.

    The tokenizer's own end-of-text, plus the chat turn-ender when there is
    one. The instruction-tuned checkpoint ends a turn with <|im_end|> and its
    weights are trained to emit it; the base checkpoint does not, and
    generation runs to the token cap instead.

    A generation_config saved with the adapter carries the same list, so
    this is a belt-and-braces check that also works with an adapter that
    was trained before the config was written.
    """
    ids: list[int] = []
    for tid in (tok.eos_token_id,):
        if isinstance(tid, int):
            ids.append(tid)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end not in ids:
        ids.append(im_end)
    if not ids:
        return None
    return ids


def generate(tok, model, questions: list[str], max_new_tokens: int = 1024,
             temperature: float = 0.0, batch_size: int = 8,
             label: str = "generating") -> list[str]:
    """Generate one completion per question.

    Reports progress. A run over six hundred problems takes a couple of hours
    and prints nothing until the end otherwise, which looks exactly like a
    hang.
    """
    outs: list[str] = []
    total = len(questions)
    t0 = time.time()
    print(f"  {label}: starting, {total} problems, {batch_size} at a time",
          flush=True)

    # A heartbeat on its own thread. A batch that takes minutes prints
    # nothing on the main thread, and a long silence is what makes a run
    # look dead. This prints the elapsed time every minute whatever the
    # batch is doing.
    stop = threading.Event()
    n_batches = (total + batch_size - 1) // batch_size
    state = {"batch": 1}
    eos = stop_token_ids(tok, model)

    def heartbeat() -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            el = time.time() - t0
            done = len(outs)
            line = progress_line(label, done, total, el,
                                 batch=state["batch"], batches=n_batches)
            print(f"{line}  (still working)", flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return _run_batches(tok, model, questions, outs, max_new_tokens,
                            temperature, batch_size, label, total, t0, state,
                            eos_token_id=eos)
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _run_batches(tok, model, questions, outs, max_new_tokens, temperature,
                 batch_size, label, total, t0, state, eos_token_id=None,
                 thinking=DEFAULT_THINKING):
    n_batches = (total + batch_size - 1) // batch_size
    for n, i in enumerate(range(0, total, batch_size)):
        state["batch"] = n + 1
        chunk = questions[i: i + batch_size]
        prompts = []
        for q in chunk:
            if isinstance(q, (tuple, list)) and len(q) == 2:
                # a (system, user) pair, rendered through the chat template
                prompts.append(render_chat(tok, q[0], q[1],
                                           thinking=thinking))
            else:
                prompts.append(build_prompt(q))
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos_token_id,
            )
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
        done = min(i + len(chunk), total)
        print(progress_line(label, done, total, time.time() - t0,
                             batch=n + 1, batches=n_batches), flush=True)
    return outs


