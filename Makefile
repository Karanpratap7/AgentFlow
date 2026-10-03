PYTHON ?= python3

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install:  ## Editable install with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

.PHONY: demo
demo:  ## End-to-end offline walkthrough
	$(PYTHON) -m agentflow.cli demo

.PHONY: test
test:  ## Run the test suite
	$(PYTHON) -m pytest

.PHONY: test-verbose
test-verbose:  ## Run the test suite with per-test output
	$(PYTHON) -m pytest -v

.PHONY: cover
cover:  ## Run tests with a coverage report
	$(PYTHON) -m pytest --cov=agentflow --cov-report=term-missing

.PHONY: lint
lint:  ## Lint and format-check
	$(PYTHON) -m ruff check agentflow tests examples

.PHONY: format
format:  ## Auto-format and fix import order
	$(PYTHON) -m ruff check --fix agentflow tests examples
	$(PYTHON) -m ruff format agentflow tests examples

.PHONY: typecheck
typecheck:  ## Static types (optional dependency)
	$(PYTHON) -m mypy agentflow || true

.PHONY: serve
serve:  ## Start the FastAPI server on :8000
	$(PYTHON) -m agentflow.cli serve --with-mcp

.PHONY: mcp
mcp:  ## Connect to every MCP server in the manifest
	$(PYTHON) -m agentflow.cli mcp

.PHONY: graph
graph:  ## Print the compiled LangGraph workflow
	$(PYTHON) -m agentflow.cli graph

.PHONY: guard
guard:  ## Run the guardrail check: make guard TEXT="..."
	$(PYTHON) -m agentflow.cli guard --text "$(TEXT)"

.PHONY: examples
examples:  ## Run every example script
	@for f in examples/*.py; do echo "== $$f"; $(PYTHON) "$$f" || exit 1; done

.PHONY: check
check: lint test  ## Lint and test

.PHONY: clean
clean:  ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info .index