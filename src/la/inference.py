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

from .schema import validate_spec


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


def build_prompt(question: str) -> str:
    return f"### Problem\n{question}\n\n### Scratchpad\n"


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


def extract_json(text: str) -> Any:
    """Pull the first balanced JSON object out of a block of text."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def parse_completion(text: str) -> dict:
    """Split a completion into its three blocks."""
    scratch = text
    answer = ""
    plot_text = ""
    m = re.search(r"###\s*Answer", text)
    if m:
        scratch = text[: m.start()]
        rest = text[m.end():]
        p = re.search(r"###\s*Plot", rest)
        if p:
            answer = rest[: p.start()]
            plot_text = rest[p.end():]
        else:
            answer = rest
    else:
        # no Answer marker: treat whatever follows the scratchpad as the answer
        m2 = re.search(r"###\s*Scratchpad", text)
        if m2:
            scratch = text[: m2.start()]
            answer = text[m2.end():]
    spec = extract_json(plot_text)
    out = {
        "scratchpad": scratch.strip(),
        "answer": answer.strip(),
        "plot_text": plot_text.strip(),
        "spec": spec,
        "spec_errors": validate_spec(spec) if spec else ["no plot spec"],
    }
    return out


def summarise_answer(text: str, limit: int = 90) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= limit else t[: limit - 1] + "…"
