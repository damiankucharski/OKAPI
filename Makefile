.PHONY: test
.PHONY: mypy
.PHONY: all_test
.PHONY: build_docs
.PHONY: serve_docs
.PHONY: deploy_docs
.PHONY: ruff

test:
	@echo "Running tests..."
	uv run pytest test

mypy:
	@echo "Running mypy..."
	uv run mypy okapi

build_docs:
	@echo "Building docs..."
	uv run mkdocs build

serve_docs:
	@echo "Serving docs..."
	uv run mkdocs serve

deploy_docs:
	@echo "Deploying docs..."
	uv run mkdocs gh-deploy

ruff:
	@echo "Running ruff..."
	uv run ruff check okapi --fix


test_all: test mypy


