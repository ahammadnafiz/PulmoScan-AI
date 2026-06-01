# PulmoScan AI — end-to-end dev orchestration
#
#   make up     # build + start the API in Docker (detached), wait until healthy
#   make web    # start the Next.js frontend on :3000  → upload + test
#   make down   # stop the API
#
.PHONY: help up down restart logs web web-install build-web mlflow clean
.DEFAULT_GOAL := help

API_URL ?= http://localhost:8000
WEB_DIR  := web

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-13s\033[0m %s\n",$$1,$$2}'

up: ## Build + start the API (Docker, detached) and wait for it to be healthy
	docker compose up -d --build
	@echo "⏳ waiting for the API to load the model..."
	@until curl -fsS $(API_URL)/api/v1/health/live >/dev/null 2>&1; do sleep 1; done
	@until curl -fsS $(API_URL)/api/v1/health/ready >/dev/null 2>&1; do sleep 1; done
	@echo "✅ API ready → $(API_URL)   (docs: $(API_URL)/docs)"

down: ## Stop the API container
	docker compose down

restart: down up ## Restart the API

logs: ## Tail API logs
	docker compose logs -f

web-install: ## Install web dependencies if missing
	@cd $(WEB_DIR) && [ -d node_modules ] || npm install

web: web-install ## Start the Next.js dev server on :3000
	@echo "🌐 starting web UI → http://localhost:3000"
	cd $(WEB_DIR) && npm run dev

build-web: web-install ## Production build of the web app
	cd $(WEB_DIR) && npm run build

mlflow: ## Launch the MLflow UI → http://localhost:5050
	@echo "📊 starting MLflow UI → http://localhost:5050"
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050 --host 0.0.0.0 --allowed-hosts "localhost:5050,127.0.0.1:5050,localhost,127.0.0.1"

clean: down ## Stop the API and remove web build artifacts
	rm -rf $(WEB_DIR)/.next $(WEB_DIR)/node_modules
