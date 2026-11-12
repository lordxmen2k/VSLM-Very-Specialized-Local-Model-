# VSLM — a linear algebra model for one 12 GB card.
#
#   make setup     install the package and the training stack
#   make setup-gpu install a CUDA-matched torch and bitsandbytes (Windows)
#   make preflight check the GPU, CUDA and bitsandbytes before training
#   make data      build the corpus and the held-out set
#   make sample    render a sample of figures as a sanity check
#   make train     fine-tune with QLoRA
#   make eval      score the model on the held-out set
#   make bench     score it against the published LinAlg-Bench results
#   make chat      ask it something
#   make test      run the unit tests
#
# Override anything on the command line, e.g.
#   make train TRAIN_ARGS="--epochs 3 --lr 1.5e-4"

PYTHON ?= python
N_TRAIN ?= 20000
N_TEST ?= 200
SEED ?= 7
WORKERS ?= 4
BASE ?= Qwen/Qwen3-1.7B-Base
ADAPTER ?= runs/vslm
DATA ?= data

export PYTHONPATH := src

.PHONY: help setup setup-gpu setup-light preflight data sample train eval bench bench-all chat test clean figures

help:
	@echo "VSLM"
	@echo "  make setup-gpu    install a CUDA-matched torch and bitsandbytes (Windows)"
	@echo "  make setup       install the package and the training stack"
	@echo "  make preflight   check the GPU, CUDA and bitsandbytes before training"
	@echo "  make data        build the corpus and the held-out set"
	@echo "  make sample      render a sample of figures as a sanity check"
	@echo "  make train       fine-tune with QLoRA"
	@echo "  make eval        score the model on the held-out set"
	@echo "  make bench       score it against the published LinAlg-Bench results"
	@echo "  make chat        ask it something"
	@echo "  make test        run the unit tests"

setup:
	$(PYTHON) -m pip install -e ".[train,dev]"

# install a CUDA-matched torch and bitsandbytes. Windows only: the script
# picks the index that matches your GPU and stops pip from replacing torch
# with the PyPI build.
setup-gpu:
	@cmd //c scripts\\setup_windows_gpu.bat

setup-light:
	$(PYTHON) -m pip install -e .

# check the machine before asking it to train
preflight:
	$(PYTHON) -m la.preflight

data:
	$(PYTHON) -m la.build_dataset --n-train $(N_TRAIN) --n-test $(N_TEST) \
		--seed $(SEED) --out $(DATA) --workers $(WORKERS)

sample:
	$(PYTHON) -m la.build_dataset --n-train 240 --n-test 24 --seed $(SEED) \
		--out $(DATA)-sample --workers $(WORKERS) --render-first 24

train:
	$(PYTHON) -m la.train --base $(BASE) --data $(DATA)/train.jsonl --out $(ADAPTER)

eval:
	$(PYTHON) -m la.evaluate --base $(BASE) --adapter $(ADAPTER) \
		--data $(DATA)/test.jsonl --out $(DATA)/results.jsonl

# the published comparison: LinAlg-Bench, with the base model as a baseline
bench:
	$(PYTHON) -m la.bench --base $(BASE) --adapter $(ADAPTER) --benchmark linalg \
		--out $(DATA)/bench_linalg.jsonl

bench-all:
	$(PYTHON) -m la.bench --base $(BASE) --adapter $(ADAPTER) --benchmark linalg \
		--baseline --perf --out $(DATA)/bench_linalg.jsonl

chat:
	$(PYTHON) -m la.chat --base $(BASE) --adapter $(ADAPTER)

test:
	$(PYTHON) -m unittest discover -s tests -v

clean:
	rm -rf runs/ outputs/ .pytest_cache/
	find . -name '__pycache__' -type d -exec rm -rf {} +
