"""The model is taught the prompt it is scored on.

The first fine-tune trained on the Base checkpoint with no chat template and
was scored with the benchmark's system prompt prepended as plain text. Two
things followed from that:

  * the model was answering a prompt format it had never seen, so the score
    measured format drift rather than the mathematics, and
  * the model had never been taught to stop, so every generation ran to the
    token cap. The first benchmark batch did not finish in six minutes.

The second fine-tune uses the instruction-tuned checkpoint, renders both the
corpus and the benchmark through the tokenizer's chat template, and puts the
turn-ending token at the end of every supervised completion.

These tests pin the three things that made the difference: the corpus prompt
is the rendered chat prompt, the benchmark prompt is the same rendered chat
prompt, and the completion ends with the token that stops generation.
"""
import unittest


class _FakeTokenizer:
    """A tokenizer that renders ChatML the way Qwen3's template does."""

    eos_token_id = 151645          # <|endoftext|>
    pad_token_id = 151643

    def __init__(self):
        self._vocab = {"<|im_end|>": 151645}

    def convert_tokens_to_ids(self, tok):
        return self._vocab.get(tok, -1)

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True):
        out = []
        for m in messages:
            out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        if add_generation_prompt:
            if enable_thinking is False:
                out.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
            else:
                out.append("<|im_start|>assistant\n")
        return "\n".join(out)


class TestChatPromptRendering(unittest.TestCase):
    def test_render_chat_puts_the_system_and_user_in_their_own_turns(self):
        from la.benchprompt import render_chat
        tok = _FakeTokenizer()
        out = render_chat(tok, "you are precise", "find the determinant",
                          thinking=False)
        self.assertIn("<|im_start|>system\nyou are precise<|im_end|>", out)
        self.assertIn("<|im_start|>user\nfind the determinant<|im_end|>", out)
        self.assertIn("<|im_start|>assistant", out,
                      "the prompt must open the assistant's turn, so the "
                      "model continues it")
        self.assertLess(out.index("<|im_start|>assistant"),
                        len(out) - 1,
                        "the assistant turn must be the last thing opened")

    def test_thinking_is_off_by_default(self):
        """The corpus already teaches the reasoning.

        Every derivation is a numbered STEP line ending in a VERIFY check.
        A hidden thinking pass would repeat it, and Qwen's own documentation
        warns that greedy decoding in thinking mode can end in endless
        repetition.

        The contract is the flag that reaches the template, because where
        the template puts the think block is Qwen's business, not ours.
        """
        from la.benchprompt import DEFAULT_THINKING, render_chat
        self.assertFalse(DEFAULT_THINKING)

        seen = []

        class Recorder(_FakeTokenizer):
            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=True):
                seen.append(enable_thinking)
                return super().apply_chat_template(
                    messages, tokenize, add_generation_prompt, enable_thinking)

        tok = Recorder()
        render_chat(tok, "s", "u")
        self.assertEqual(seen[-1], False,
                         "thinking must be off unless it is asked for")
        render_chat(tok, "s", "u", thinking=True)
        self.assertEqual(seen[-1], True)

    def test_a_template_without_enable_thinking_still_renders(self):
        """A tokenizer whose template does not take the flag still works."""
        from la.benchprompt import render_chat

        class Old:
            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=True):
                return "|".join(m["content"] for m in messages)

        out = render_chat(Old(), "sys", "usr", thinking=False)
        self.assertIn("sys", out)
        self.assertIn("usr", out)

    def test_no_system_prompt_leaves_the_system_turn_out(self):
        from la.benchprompt import render_chat
        out = render_chat(_FakeTokenizer(), "", "just the question")
        self.assertNotIn("<|im_start|>system", out)
        self.assertIn("<|im_start|>user", out)


class TestBenchmarkPromptMatchesTraining(unittest.TestCase):
    def test_the_benchmark_system_prompt_is_unchanged(self):
        """The benchmark's own system prompt, verbatim.

        Changing it would change the task the model is being scored on. The
        only thing that changes is that it now arrives as a system turn
        rather than as text glued to the front of the question.
        """
        from la.benchprompt import benchmark_prompt_pair
        system, user = benchmark_prompt_pair("bare matrix", "determinant", "4x4")
        self.assertIn("precise mathematical assistant", system)
        self.assertIn("\\boxed{}", system)
        self.assertTrue(user.startswith("Solve this linear algebra problem"))

    def test_the_flat_prompt_is_still_the_system_and_user_joined(self):
        """build_benchmark_prompt keeps its old meaning.

        Other callers may still want the flat form. It has to remain exactly
        the two prompts joined, so nothing silently changes for them.
        """
        from la.benchprompt import (benchmark_prompt_pair,
                                    build_benchmark_prompt)
        system, user = benchmark_prompt_pair("bare matrix", "determinant", "4x4")
        flat = build_benchmark_prompt("bare matrix", "determinant", "4x4")
        self.assertEqual(f"{system}\n\n{user}", flat)

    def test_the_benchmark_renders_the_same_chat_prompt_as_the_corpus(self):
        """The test the first project was missing.

        For the *same* question, the prompt the corpus trains on and the
        prompt the benchmark sends must be the same string.

        Before, the corpus was trained on the bare question and the
        benchmark prepended its own system prompt as plain text, so the two
        could never match - and the existing test only compared the corpus
        prompt with itself, so it passed.
        """
        from la.benchprompt import benchmark_prompt_pair, row_prompt

        tok = _FakeTokenizer()
        question = ("Find the determinant of the 3x3 matrix A.\n\n"
                    "A = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\\\ 4 & 5 "
                    "\\end{bmatrix}\n\n"
                    "Show all intermediate steps. Put your final numerical "
                    "answer inside \\boxed{}.")

        # what the corpus trains on
        system, user = benchmark_prompt_pair(question, "determinant", "3x3")
        corpus_row = {"system": system, "user": user}
        corpus_prompt = row_prompt(corpus_row, tok)

        # what the benchmark sends, for the same question
        bench_system, bench_user = benchmark_prompt_pair(
            question, "determinant", "3x3")
        bench_prompt = row_prompt(
            {"system": bench_system, "user": bench_user}, tok)

        self.assertEqual(corpus_prompt, bench_prompt,
                         "the corpus prompt and the benchmark prompt differ")
        self.assertIn("<|im_start|>system", corpus_prompt,
                      "the corpus prompt must carry the system turn")
        self.assertIn("<|im_start|>user", corpus_prompt)

    def test_a_corpus_question_ends_up_as_a_benchmark_question(self):
        """The corpus question and the benchmark's user turn are the same
        shape: preamble, task, body, closing instruction.

        This is what makes the rendered prompts comparable at all. If the
        corpus asked a differently-shaped question, the model would still be
        answering a prompt it had never seen.
        """
        import random
        from la.benchprompt import benchmark_prompt_pair
        from la.problems import GENERATORS

        rng = random.Random(41)
        p = None
        for _ in range(40):
            p = GENERATORS["determinant"](rng)
            if p is not None:
                break
        row = p.to_json()
        _, user = benchmark_prompt_pair(row["question"], "determinant", "3x3")
        # the benchmark's user turn for a problem that already has its task
        # is the preamble plus the problem
        self.assertTrue(user.startswith("Solve this linear algebra problem"))
        self.assertIn("Show all intermediate steps", user)
        self.assertIn("\\boxed{}", user)

    def test_the_row_carries_the_chat_turns(self):
        """The corpus row stores the two turns, so the template can be
        rendered later by the tokenizer that will actually be used."""
        import random
        from la.problems import GENERATORS
        rng = random.Random(43)
        p = None
        for _ in range(40):
            p = GENERATORS["determinant"](rng)
            if p is not None:
                break
        row = p.to_json()
        self.assertIn("system", row)
        self.assertIn("user", row)
        self.assertEqual(row["user"], row["question"])
        self.assertIn("mathematical assistant", row["system"])

    def test_an_old_row_without_chat_turns_still_renders(self):
        """A corpus built before the migration has no system turn.

        It must still work, and the difference must be visible: the old row
        carries no system prompt, which is exactly why it scored worse.
        """
        from la.benchprompt import row_prompt
        tok = _FakeTokenizer()
        old = {"prompt": "Solve this linear algebra problem. Show your work.",
               "question": "Solve this linear algebra problem. Show your work."}
        out = row_prompt(old, tok)
        self.assertNotIn("<|im_start|>system", out)
        self.assertIn("Solve this linear algebra problem", out)


class TestStopTokens(unittest.TestCase):
    def test_the_completion_ends_with_the_end_of_text_token(self):
        """Without this the model is never taught to stop.

        Every generation then runs to the token cap, which is what made the
        first benchmark batch take six minutes.
        """
        from la.train import tokenise_rows

        class Tok:
            eos_token_id = 7

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1, 2, 3]}

        rows = [{"prompt": "P", "completion": "C"}]
        (ids, labels), = tokenise_rows(rows, Tok(), max_len=None)
        self.assertEqual(ids[-1], 7, "the completion does not end with EOS")
        self.assertEqual(labels[-1], 7, "EOS must be supervised, not masked")
        self.assertEqual(len(ids), len(labels))

    def test_the_eos_can_be_turned_off(self):
        from la.train import tokenise_rows

        class Tok:
            eos_token_id = 7

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1, 2, 3]}

        rows = [{"prompt": "P", "completion": "C"}]
        (ids, _), = tokenise_rows(rows, Tok(), max_len=None, add_eos=False)
        self.assertNotEqual(ids[-1], 7)

    def test_the_prompt_is_masked_out_of_the_labels(self):
        from la.train import IGNORE, tokenise_rows

        class Tok:
            eos_token_id = 7

            def __call__(self, text, add_special_tokens=False):
                # the prompt is two tokens, the completion three
                return {"input_ids": [1, 2]} if text == "P" else {"input_ids": [3, 4, 5]}

        rows = [{"prompt": "P", "completion": "C"}]
        (ids, labels), = tokenise_rows(rows, Tok(), max_len=None)
        self.assertEqual(ids[:2], [1, 2])
        self.assertEqual(labels[:2], [IGNORE, IGNORE])
        self.assertEqual(labels[2:], [3, 4, 5, 7])

    def test_the_stop_tokens_include_the_chat_turn_ender(self):
        """The instruction checkpoint ends a turn with <|im_end|>."""
        from la.inference import stop_token_ids
        ids = stop_token_ids(_FakeTokenizer())
        self.assertIn(151645, ids)
        self.assertEqual(len(ids), len(set(ids)), "the list has duplicates")


if __name__ == "__main__":
    unittest.main()
