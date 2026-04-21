# EdgeGrid Forecast Engine — dev & deploy helper
# ------------------------------------------------
# Usage:
#   make install         # pip install -e ".[api,dev]"
#   make test            # run pytest
#   make lint            # ruff + mypy
#   make run             # serve locally (uvicorn --reload)
#   make benchmark       # run all 4 validation protocols
#   make docker-build    # build production image
#   make docker-run      # run image on :8000
#   make systemd-install # copy unit file to /etc/systemd/system/

PY          ?= python3
PIP         ?= $(PY) -m pip
PYTEST      ?= $(PY) -m pytest
RUFF        ?= $(PY) -m ruff
UVICORN     ?= $(PY) -m uvicorn
IMAGE       ?= edgegrid-forecast
TAG         ?= v5
APP         ?= edgegrid_forecast.api.main:app
HOST        ?= 0.0.0.0
PORT        ?= 8000

.PHONY: help install install-dev test lint format clean \
        run run-prod benchmark-s1 benchmark-s2 benchmark-s3 benchmark-forward benchmark \
        docker-build docker-run docker-push systemd-install

help:
	@echo "EdgeGrid Forecast Engine — available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ─── Install ───────────────────────────────────────────────────────────────
install: ## Install runtime deps (editable)
	$(PIP) install -e ".[api]"

install-dev: ## Install runtime + dev deps
	$(PIP) install -e ".[api,dev]"

# ─── Quality gates ─────────────────────────────────────────────────────────
test: ## Run pytest suite
	PYTHONPATH=src $(PYTEST) -q tests/

lint: ## Ruff lint
	$(RUFF) check src/ tests/ benchmarks/

format: ## Ruff format (auto-fix)
	$(RUFF) format src/ tests/ benchmarks/
	$(RUFF) check --fix src/ tests/ benchmarks/

clean: ## Remove caches / build artefacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete

# ─── Serve ─────────────────────────────────────────────────────────────────
run: ## Dev server (auto-reload)
	PYTHONPATH=src $(UVICORN) $(APP) --host $(HOST) --port $(PORT) --reload

run-prod: ## Prod server (1 worker, no reload)
	PYTHONPATH=src $(UVICORN) $(APP) --host $(HOST) --port $(PORT) --workers 1 --log-level info

# ─── Benchmarks (v5 validation protocols) ──────────────────────────────────
benchmark-s1: ## S1 chronological holdout — model quality
	PYTHONPATH=src $(PY) benchmarks/v5_benchmark_s1.py

benchmark-s2: ## S2 stratified — every-4th-day holdout
	PYTHONPATH=src $(PY) benchmarks/v5_benchmark_s2.py

benchmark-s3: ## S3 rolling-origin — 7-day forward (light/leaky)
	PYTHONPATH=src $(PY) benchmarks/v5_benchmark_s3.py

benchmark-forward: ## Forward Apr 21 → May 20 — 4 strategies
	PYTHONPATH=src $(PY) benchmarks/forward_v5_feb12_strategies.py

benchmark: benchmark-s1 benchmark-s2 benchmark-s3 benchmark-forward ## Run all 4 protocols

# ─── Docker ────────────────────────────────────────────────────────────────
docker-build: ## Build production image
	docker build -t $(IMAGE):$(TAG) -t $(IMAGE):latest .

docker-run: ## Run image on :$(PORT)
	docker run --rm -p $(PORT):8000 \
	  -v $(PWD)/models:/app/models:ro \
	  -v $(PWD)/data:/app/data:ro \
	  --name edgegrid-forecast-$(TAG) \
	  $(IMAGE):$(TAG)

docker-push: ## Push image (override REGISTRY=... IMAGE=...)
	docker tag $(IMAGE):$(TAG) $(REGISTRY)/$(IMAGE):$(TAG)
	docker push $(REGISTRY)/$(IMAGE):$(TAG)

# ─── systemd (bare-metal / VM deploy) ──────────────────────────────────────
systemd-install: ## Install systemd unit (run as root)
	install -m 0644 deploy/edgegrid-forecast.service /etc/systemd/system/edgegrid-forecast.service
	systemctl daemon-reload
	@echo "Enable with: systemctl enable --now edgegrid-forecast"
