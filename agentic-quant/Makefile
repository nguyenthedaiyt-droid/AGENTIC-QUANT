.PHONY: help install install-dev test lint format clean run test-unit test-integration test-coverage test-slow install-precommit setup

# =============================================================================
# AGENTIC-QUANT — Development Makefile
# Quy uoc: make <target> [VAR=value]
# Vi du: make install PYTHON_VERSION=3.11
# =============================================================================

help:  ## Hien thi danh sach cac lenh co san
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-20s\033[0m %s\n", $$1, $$2}'

# -----------------------------------------------------------------------------
# cai dat
# -----------------------------------------------------------------------------

install:  ## Cai dat production dependencies
	poetry install --no-interaction

install-dev:  ## Cai dat development dependencies (includes test, linting)
	poetry install --with dev,test --no-interaction

install-precommit:  ## cai dat pre-commit hooks
	poetry run pre-commit install --install-hooks

setup: install-dev install-precommit  ## Full setup: install deps + pre-commit

# -----------------------------------------------------------------------------
# testing
# -----------------------------------------------------------------------------

test:  ## Chay tat ca tests (unit + integration)
	poetry run pytest $(PYTEST_ARGS)

test-unit:  ## Chi chay unit tests
	poetry run pytest tests/unit $(PYTEST_ARGS)

test-integration:  ## Chi chay integration tests
	poetry run pytest tests/integration $(PYTEST_ARGS)

test-slow:  ## Chay tests nhanh (bo qua @pytest.mark.slow)
	poetry run pytest -m "not slow" $(PYTEST_ARGS)

test-coverage:  ## Chay tests voi bao cao coverage HTML
	poetry run pytest --cov=core --cov-report=html --cov-report=term-missing

test-verbose:  ## Chay tests voi output chi tiet
	poetry run pytest -v --tb=long $(PYTEST_ARGS)

# -----------------------------------------------------------------------------
# linting & formatting
# -----------------------------------------------------------------------------

lint:  ## Chay tat ca linting tools (ruff, mypy)
	poetry run ruff check core/
	poetry run ruff check tests/
	poetry run mypy core/

format:  ## Format tat ca code (black + ruff)
	poetry run black core/ tests/ scripts/
	poetry run ruff check --fix core/ tests/ scripts/

# -----------------------------------------------------------------------------
# chay ung dung
# -----------------------------------------------------------------------------

run:  ## Khoi dong backend server
	poetry run python -m core.main

run-backtest:  ## Chay backtest
	poetry run python scripts/run_backtest.py $(BACKTEST_ARGS)

# -----------------------------------------------------------------------------
# database
# -----------------------------------------------------------------------------

db-migrate:  ## Chay SQLite migrations
	poetry run python -m scripts.migrations.run

db-reset:  ## Xoa va tao lai database (WARNING: mat du lieu!)
	rm -f data/agentic_quant.db
	$(MAKE) db-migrate

# -----------------------------------------------------------------------------
# docker
# -----------------------------------------------------------------------------

docker-up:  ## Khoi dong Redis + Qdrant qua Docker Compose
	docker compose up -d

docker-down:  ## Dung Docker Compose services
	docker compose down

docker-logs:  ## Xem logs cua Docker Compose
	docker compose logs -f

# -----------------------------------------------------------------------------
# model training (requires data)
# -----------------------------------------------------------------------------

train-lstm:  ## Huan luyen LSTM Autoencoder
	poetry run python scripts/train_lstm.py $(TRAIN_ARGS)

train-model-a:  ## Huan luyen XGBoost Model A
	poetry run python scripts/train_model_a.py $(TRAIN_ARGS)

train-model-b:  ## Huan luyen XGBoost Model B
	poetry run python scripts/train_model_b.py $(TRAIN_ARGS)

# -----------------------------------------------------------------------------
# don dep
# -----------------------------------------------------------------------------

clean:  ## Don dep build artifacts, cache, coverage reports
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -rf .coverage
	rm -rf __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/
	rm -rf build/
	rm -rf ui/dist/
	rm -rf ui/node_modules/

# -----------------------------------------------------------------------------
# phan quyen
# -----------------------------------------------------------------------------

chmod-scripts:
	@chmod +x scripts/*.sh 2>/dev/null || echo "chmod skipped (not Unix)"
