"""A faithful-enough Qwen3 BPE tokenizer, for token counting.

The sandbox has no `transformers` and no PyPI access, but the corpus's
length budget has to be measured in REAL tokens, not in characters
divided by a heuristic. The character heuristic (divide by 2.04) is what
made the 5x5 exception look necessary when it was not.

This implements the tokenizer's own pipeline from `tokenizer.json`:

    normalizer   NFC
    pre_tokenizer  the Qwen regex split, then ByteLevel
    model        BPE with the vocabulary and the merges

Byte-level BPE, so every byte has a character and every merge is a pair
of symbols.

The measurement it exists for:

    matrix_power 5x5 with the full dot product   worst 840 tokens
    matrix_multiply 5x5 with the full dot product worst 880 tokens
    the training cap                                2048 tokens

Both fit. The 5x5 exception was costing the matrix_power family 26
problems (41/60 to 15/60) and it was never needed.
"""
from __future__ import annotations

import heapq
import json
import os
import re
import unicodedata

# The tokenizer, fetched from the model's own repository:
#   https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/tokenizer.json
#
# It is NOT committed - 11 MB of vocabulary is not worth a repository -
# so `make tokenizer` fetches it into .cache/ and the count falls back
# to a character estimate if it is missing.
TOKENIZER_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".cache", "qwen3_tokenizer.json")

def _is_letter(ch: str) -> bool:
    return ch.isalpha()


def _is_number(ch: str) -> bool:
    return ch.isdigit()


def _is_space(ch: str) -> bool:
    return ch.isspace() or ch in "\r\n"


def _split_pre(text: str):
    """The Qwen pre-tokenizer's split, done by hand.

    Python's `re` has no `\\p{L}`, so the regex is reproduced as a
    scanner:

        (?i:'s|'t|'re|'ve|'m|'ll|'d)
      | [^\\r\\n\\p{L}\\p{N}]?\\p{L}+
      | \\p{N}
      | ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*
      | \\s*[\\r\\n]+
      | \\s+(?!\\S)
      | \\s+
    """
    contractions = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        # the contractions, case-insensitively
        matched = False
        for c in contractions:
            if text[i:i + len(c)].lower() == c:
                out.append(text[i:i + len(c)])
                i += len(c)
                matched = True
                break
        if matched:
            continue
        # \s*[\r\n]+
        if _is_space(ch) and ch != "\r" and ch != "\n":
            j = i
            while j < n and _is_space(text[j]) and text[j] not in "\r\n":
                j += 1
            if j < n and text[j] in "\r\n":
                k = j
                while k < n and text[k] in "\r\n":
                    k += 1
                out.append(text[i:k])
                i = k
                continue
        if ch == "\r" or ch == "\n":
            k = i
            while k < n and text[k] in "\r\n":
                k += 1
            out.append(text[i:k])
            i = k
            continue
        # [^\r\n\p{L}\p{N}]?\p{L}+   - an optional prefix then letters
        j = i
        prefix = ""
        if not _is_letter(ch) and not _is_number(ch) and ch not in "\r\n":
            j = i + 1
            prefix = ch
        if j < n and _is_letter(text[j]):
            k = j
            while k < n and _is_letter(text[k]):
                k += 1
            out.append(prefix + text[j:k])
            i = k
            continue
        # \p{N}
        if _is_number(ch):
            k = i
            while k < n and _is_number(text[k]):
                k += 1
            out.append(text[i:k])
            i = k
            continue
        # ?[^\s\p{L}\p{N}]+[\r\n]*
        if not _is_space(ch) and not _is_letter(ch) and not _is_number(ch):
            k = i
            if text[k] == " ":
                k += 1
            while k < n and not _is_space(text[k]) and not _is_letter(text[k]) \
                    and not _is_number(text[k]):
                k += 1
            while k < n and text[k] in "\r\n":
                k += 1
            out.append(text[i:k])
            i = k
            continue
        # \s+(?!\S)  and  \s+
        j = i
        while j < n and _is_space(text[j]):
            j += 1
        if j < n and j - 1 > i:
            # back off one so the last space can join the next token.
            # The guard matters: a ONE-character whitespace run followed
            # by a non-space would otherwise leave j == i, and the scan
            # would never advance. It hung on a 20-line derivation.
            j -= 1
        out.append(text[i:j])
        i = j
    return out


def _bytes_to_unicode():
    """The byte-level alphabet: reversible map from bytes to characters."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class QwenBPE:
    """A byte-level BPE over a HF `tokenizer.json`."""

    def __init__(self, path: str = TOKENIZER_JSON):
        with open(path, encoding="utf-8") as fh:
            spec = json.load(fh)
        model = spec["model"]
        self.vocab = model["vocab"]
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # the merges, in order
        self.ranks = {}
        for i, merge in enumerate(model["merges"]):
            if isinstance(merge, str):
                a, b = merge.split(" ")
            else:
                a, b = merge
            self.ranks[(a, b)] = i
        self.specials = {t["content"]: t["id"] for t in spec.get("added_tokens", [])
                         if t.get("special")}
        # The BPE result of a pre-token depends only on the pre-token,
        # and a derivation repeats the same pre-tokens many times - the
        # numbers, the operators, the words. Without the cache a single
        # 5x5 example takes minutes; with it, seconds.
        self._cache: dict[str, list[str]] = {}

    # -- the pieces ----------------------------------------------------
    def _byte_encode(self, text: str) -> str:
        return "".join(self.byte_encoder[b] for b in text.encode("utf-8"))

    def _bpe(self, token: str) -> list[str]:
        """The BPE merges over one pre-token.

        The standard algorithm, with the merges looked up by RANK so the
        cheapest pair is always taken. A naive version that scans for the
        best pair from the start every round is O(n^2) per token and a
        whole derivation takes minutes; this one is a heap.

        The result is CACHED: it depends only on the pre-token, and a
        derivation repeats the same pre-tokens many times.
        """
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        symbols = list(token)
        if len(symbols) < 2:
            self._cache[token] = symbols
            return symbols
        # the heap of candidate merges, keyed by rank
        heap = []
        for i in range(len(symbols) - 1):
            rank = self.ranks.get((symbols[i], symbols[i + 1]))
            if rank is not None:
                heapq.heappush(heap, (rank, i))
        while heap:
            rank, i = heapq.heappop(heap)
            if i + 1 >= len(symbols):
                continue
            a, b = symbols[i], symbols[i + 1]
            if self.ranks.get((a, b)) != rank:
                continue  # stale entry: the symbols have changed
            symbols[i:i + 2] = [a + b]
            # the neighbours of the merged symbol may now merge
            for j in (i - 1, i):
                if 0 <= j < len(symbols) - 1:
                    r = self.ranks.get((symbols[j], symbols[j + 1]))
                    if r is not None:
                        heapq.heappush(heap, (r, j))
        self._cache[token] = symbols
        return symbols

    def encode(self, text: str) -> list[int]:
        """The token ids of a string."""
        text = unicodedata.normalize("NFC", text)
        ids = []
        # the special tokens are matched first, as HF's added-token
        # splitting does
        parts = re.split(r"(<\|[^|]*\|>)", text)
        for part in parts:
            if not part:
                continue
            if part in self.specials:
                ids.append(self.specials[part])
                continue
            for chunk in _split_pre(part):
                if not chunk:
                    continue
                encoded = self._byte_encode(chunk)
                for sym in self._bpe(encoded):
                    ids.append(self.vocab.get(sym, self.vocab.get(
                        self.byte_encoder[ord("?")], 0)))
        return ids

    def count(self, text: str) -> int:
        return len(self.encode(text))
