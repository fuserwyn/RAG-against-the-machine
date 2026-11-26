# On a 42 computer, .school.env (written by `make school`) points uv and
# Hugging Face at sgoinfre. It is optional everywhere else.
-include .school.env

# Usage: make <rule> [BACKEND=native|langchain]
#                    [MODE=lexical|semantic|hybrid] [K=10] [CHUNK=2000]
BACKEND ?= native
MODE ?= lexical
K ?= 10
CHUNK ?= 2000
PORT ?= 8000
SCOPE ?= UnansweredQuestions
PY = uv run python
CLI = $(PY) -m src --backend $(BACKEND) --mode $(MODE)
DATASETS = data/datasets
OUT = data/output/$(BACKEND)-$(MODE)
INDEX_FLAGS = $(if $(filter semantic hybrid,$(MODE)),--semantic,)

.PHONY: install run debug clean lint lint-strict index update search \
	evaluate answer compare serve test test-fast school purge \
	local up down logs rebuild ps

install:
	uv sync

## 42 computers: clone, venv and model weights on sgoinfre (scripts/school_setup.sh)
school:
	./scripts/school_setup.sh

## 42 computers, after the defense: venv, corpus and caches. The clone stays.
purge:
	./scripts/school_setup.sh --uninstall

## index + search both public datasets + self-evaluation
run: index search evaluate

index:
	$(CLI) index --max_chunk_size $(CHUNK) $(INDEX_FLAGS)

## re-index only the files changed since the last build (bonus)
update:
	$(CLI) index --max_chunk_size $(CHUNK) $(INDEX_FLAGS) --incremental

search:
	@for ds in code docs; do \
		$(CLI) search_dataset \
			--dataset_path $(DATASETS)/$(SCOPE)/dataset_$${ds}_public.json \
			--k $(K) --save_directory $(OUT)/search_results/$(SCOPE); \
	done

evaluate:
	@for ds in code docs; do \
		echo "== $(BACKEND) / $$ds"; \
		$(CLI) evaluate \
			--student_search_results_path \
				$(OUT)/search_results/$(SCOPE)/dataset_$${ds}_public.json \
			--dataset_path $(DATASETS)/AnsweredQuestions/dataset_$${ds}_public.json; \
	done

answer:
	@for ds in code docs; do \
		$(CLI) answer_dataset \
			--student_search_results_path \
				$(OUT)/search_results/$(SCOPE)/dataset_$${ds}_public.json \
			--save_directory $(OUT)/search_results_and_answer/$(SCOPE); \
	done

## run the retrieval pipeline with both backends
compare:
	$(MAKE) run BACKEND=native
	$(MAKE) run BACKEND=langchain

## local HTTP API on http://127.0.0.1:$(PORT) (bonus)
serve:
	$(CLI) serve --port $(PORT)

## pipeline visualizer, no Docker: http://127.0.0.1:8080
local:
	@echo "open http://127.0.0.1:8080"
	PYTHONPATH=. $(PY) visualizer/engine/trace_server.py \
		--host 127.0.0.1 --port 8080 --static visualizer/viz

## same visualizer in Docker (compose file stays in visualizer/)
up:
	$(MAKE) -C visualizer up

down:
	$(MAKE) -C visualizer down

logs:
	$(MAKE) -C visualizer logs

rebuild:
	$(MAKE) -C visualizer rebuild

ps:
	$(MAKE) -C visualizer ps

## every test (corpus / model tests are skipped if data or weights are
## missing); test-fast skips them explicitly
test:
	uv run pytest

test-fast:
	uv run pytest -m "not corpus and not model"

debug:
	$(PY) -m pdb -m src --backend $(BACKEND) index --max_chunk_size $(CHUNK)

clean:
	find . -type d \( -name __pycache__ -o -name .mypy_cache \
		-o -name .pytest_cache \) -not -path './.venv/*' -prune \
		-exec rm -rf {} +

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict --ignore-missing-imports
