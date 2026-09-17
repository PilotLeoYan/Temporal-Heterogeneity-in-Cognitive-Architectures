.PHONY: run format lint check

run:
	uv run python -m src

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix
	uv run mypy .

check:
	uv run pre-commit run --all-files
