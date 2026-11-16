"""Show the progress output without a GPU.

Runs the real generate() against a stub model, so you can see what a long
run prints. Not a test: it replaces torch with a stub, which would shadow
the real torch on a machine that has it.

    python scripts/demo_progress.py
"""
import sys, types, time, importlib

torch = types.ModuleType("torch")
torch.no_grad = lambda: __import__("contextlib").nullcontext()
torch.bfloat16 = 'bf16'
torch.float16 = 'fp16'
torch.float32 = 'fp32'
sys.modules["torch"] = torch

sys.path.insert(0, "src")
la_inf = importlib.import_module("la.inference")
la_inf.HEARTBEAT_SECONDS = 2          # pretend a minute is two seconds


class FakeEnc(dict):
    def to(self, device):
        return self


class FakeTok:
    pad_token_id = 0
    def __call__(self, prompts, **kw):
        class Enc:
            input_ids = types.SimpleNamespace(shape=(len(prompts), 10))
        return FakeEnc(input_ids=Enc.input_ids)
    def decode(self, ids, **kw):
        return "### Answer\n\\boxed{1}"


class FakeModel:
    device = "cpu"
    def generate(self, **kw):
        time.sleep(1.5)               # a slow batch
        return [[0] * 10 for _ in range(kw["input_ids"].shape[0])]


print("--- 24 problems, batch 8, 2-second heartbeat ---")
outs = la_inf.generate(FakeTok(), FakeModel(), [f"q{i}" for i in range(24)],
                        batch_size=8, label="demo")
print(f"--- finished, {len(outs)} completions ---")
