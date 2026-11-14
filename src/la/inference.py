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
from typing import Any

import torch

from .parse import extract_json, parse_completion, summarise_answer


def load_model(base: str, adapter: str | None = None, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter or base, trust_remote_code=True)
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


from .benchprompt import model_prompt as build_prompt


def generate(tok, model, questions: list[str], max_new_tokens: int = 1024,
             temperature: float = 0.0, batch_size: int = 8) -> list[str]:
    """Generate one completion per question."""
    outs: list[str] = []
    for i in range(0, len(questions), batch_size):
        chunk = questions[i: i + batch_size]
        prompts = [build_prompt(q) for q in chunk]
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
            )
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
    return outs


