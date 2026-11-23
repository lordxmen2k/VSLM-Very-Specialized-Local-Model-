"""The prompt decoration must not be able to move the matrix.

THE DEFECT THIS FILE GUARDS AGAINST. The corpus's task statement and
prompt tail were drawn from the SAME random stream as the matrix they
decorate. Both are prompt decoration; the matrix is the problem.

Because they shared one stream, every change to the prompt contract -
a new tail, a new weight, a different `randrange` bound, a size-aware
table - consumed a different number of draws and silently rewrote every
matrix that came after it.

The effect is invisible in the output. The corpus still looks
plausible: the same families, the same size mix, the same magnitudes.
It is simply a DIFFERENT corpus, and no test on the output can see the
difference because nothing about the output is wrong.

Measured on the eigenvalue family, seed 4242:

    run 10 (decoration shared)   max entry in a 250-problem window 1118
    run 11 (decoration isolated) max entry in a 250-problem window 1118
    ... at 1200 problems          max entry                       1120

and every family produced a different set of matrices between the two,
because the decoration's draws landed in different places.

Run 8 did the same thing from the other direction: `draw_entry()`
changed from one draw per entry to two, which rewrote 76 per cent of
the corpus and made run 8's result unreadable. That one was caught
because the corpus's SHAPE changed. This one does not change the shape,
so it has to be caught by construction.

THE FIX. The decoration draws from a stream DERIVED from the problem's
stream (`benchprompt_tails.decoration_rng`). The corpus is still
deterministic - the same corpus every time, for every seed - but the
decoration can no longer consume the problem's draws.
"""
from __future__ import annotations

import random
import unittest

from la.problems import generate


def _draw_count(rng):
    """How many draws the stream has made.

    `random.Random.getstate()` does not report a draw count. The
    reliable way is to wrap the stream's methods and count the calls -
    but the wrapper has to be the stream the caller actually uses, so
    the count is taken around the call itself.
    """
    return _CountingStream(rng)


class _CountingStream:
    """Wraps a stream and counts its draw calls."""

    def __init__(self, rng):
        self._rng = rng
        self.n = 0

    def __getattr__(self, name):
        attr = getattr(self._rng, name)
        if name in ("random", "randint", "randrange", "choice", "shuffle",
                    "getrandbits", "sample"):
            def wrapped(*a, **k):
                self.n += 1
                return attr(*a, **k)
            return wrapped
        return attr


def _matrix_texts(family, n=60, seed=4242):
    """The matrix blocks of the first n problems of a family."""
    import re
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = generate(family, rng)
        if p is None:
            continue
        for m in re.finditer(r"\\begin\{[bp]matrix\}(.*?)\\end\{[bp]matrix\}",
                              p.question, re.S):
            out.append(m.group(1))
    return out


class TestTheDecorationCannotMoveTheMatrix(unittest.TestCase):
    """The matrix must not depend on the prompt contract."""

    # The families whose prompt contract run 11 changed.
    FAMILIES = ("determinant", "eigenvalue", "matrix_power", "matrix_vector",
                "matrix_multiply", "matrix_trace", "matrix_transpose",
                "nullity", "rank")

    def test_the_decoration_does_not_consume_the_problems_stream(self):
        """The task statement and the tail must not draw from the matrix's
        stream.

        The check is a difference, not an absolute: the matrix is the
        same whichever tail the prompt carries.
        """
        from la import problems as P

        # The decoration takes ONE draw from the problem's stream - the
        # seed for its own stream - and no more. One draw is a constant,
        # so it cannot couple the decoration's choices to the matrix; the
        # property that matters is that the count does not change when
        # the prompt contract does.
        from la.augment import paraphrase

        for label, call in (("tail",
                             lambda r: P._tail(r, "eigenvalue", "4x4")),
                            ("task statement",
                             lambda r: paraphrase("eigenvalue", r, "4x4"))):
            with self.subTest(decoration=label):
                counts = set()
                for _ in range(20):
                    counter = _CountingStream(random.Random(20261117))
                    call(counter)
                    counts.add(counter.n)
                self.assertEqual(
                    {1}, counts,
                    f"the {label} drew {counts} times from the problem's "
                    f"stream; it must draw exactly once, for its seed, so "
                    f"that a change to the prompt contract cannot rewrite "
                    f"every matrix after it")

    def test_the_corpus_is_the_same_whatever_the_prompt_contract(self):
        """Changing the tail table must not change the matrices.

        This is the property run 10 did not have, and the reason this
        file exists: run 10's tail drew from the problem's stream, so the
        number of draws the tail made depended on the prompt contract,
        and a change to the contract rewrote every matrix after it.

        The fix makes the decoration take exactly ONE draw from the
        problem's stream - its seed - so the count cannot depend on the
        contract at all.
        """
        from la import benchprompt_tails as T
        from la import problems as P

        original = T.PROMPT_TAILS_BY_SIZE
        try:
            baseline = _matrix_texts("eigenvalue")
            # a different tail table: same shapes, different text
            T.PROMPT_TAILS_BY_SIZE = {
                sub: {size: [(m + " (extra)", c, w) for m, c, w in entries]
                      for size, entries in per_size.items()}
                for sub, per_size in original.items()
            }
            changed = _matrix_texts("eigenvalue")
        finally:
            T.PROMPT_TAILS_BY_SIZE = original

        self.assertEqual(
            baseline, changed,
            "changing the tail table changed the matrices, so the tail is "
            "still sharing the problem's random stream")

    def test_the_decoration_draw_count_does_not_depend_on_the_contract(self):
        """The strongest form of the property.

        If the decoration's draw count were a function of the prompt
        contract, then changing the contract would shift the problem's
        stream. So the count must be the same for every contract.
        """
        from la import benchprompt_tails as T
        from la import problems as P

        original = T.PROMPT_TAILS_BY_SIZE
        counts = set()
        try:
            for extra in ("", " (extra)", " (a much longer extra)"):
                T.PROMPT_TAILS_BY_SIZE = {
                    sub: {size: [(m + extra, c, w) for m, c, w in entries]
                          for size, entries in per_size.items()}
                    for sub, per_size in original.items()
                }
                counter = _CountingStream(random.Random(20261117))
                P._tail(counter, "eigenvalue", "4x4")
                counts.add(counter.n)
        finally:
            T.PROMPT_TAILS_BY_SIZE = original

        self.assertEqual(
            {1}, counts,
            f"the decoration's draw count was {counts} for different "
            f"prompt contracts; it must not depend on the contract")


if __name__ == "__main__":
    unittest.main()
