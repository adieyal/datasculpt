.PHONY: help install install-dev test lint format typecheck docs-generate docs-serve docs-clean clean

help:
	@echo "Available commands:"
	@echo "  install       Install package"
	@echo "  install-dev   Install with dev dependencies"
	@echo "  test          Run tests"
	@echo "  lint          Run linter"
	@echo "  format        Format code"
	@echo "  typecheck     Run type checker"
	@echo "  docs-generate Build documentation site"
	@echo "  docs-serve    Serve documentation locally"
	@echo "  docs-clean    Remove generated docs"
	@echo "  clean         Remove all build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs]"

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy src

# Documentation
docs-generate:
	mkdocs build

docs-serve:
	mkdocs serve

docs-clean:
	rm -rf site/

# Cleanup
clean: docs-clean
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
