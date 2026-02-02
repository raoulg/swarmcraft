# Makefile for SwarmCraft

# Use .PHONY to ensure these commands run even if a file with the same name exists.
.PHONY: help setup up down build redis test test-integration test-all test-cov clean prod

# Default command to run when you just type "make"
help:
	@echo "Available commands:"
	@echo "  setup            - Create .env and install dependencies (local dev)."
	@echo "  up               - Start dev services (api, redis) with Docker Compose."
	@echo "  prod             - Build and run production container (no local uv sync)."
	@echo "  down             - Stop and remove all services."
	@echo "  build            - Rebuild the Docker images."
	@echo "  redis            - Start only the Redis container in the background."
	@echo "  test             - Run fast, local unit tests (no Redis required)."
	@echo "  test-integration - Run integration tests (requires Redis)."
	@echo "  test-all         - Run all tests (local and integration)."
	@echo "  test-cov         - Run all tests and generate a coverage report."
	@echo "  clean            - Remove temporary Python/test files."

# Project Setup (Local Dev)
setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from example"; \
	fi
	uv sync

# Docker Compose Management (Dev)
up:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from example"; \
	fi
	COMPOSE_BAKE=true docker compose up

down:
	docker compose down

build:
	COMPOSE_BAKE=true docker compose build

# Production
prod:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from example"; \
	fi
	@echo "Building production image..."
	docker build -f Dockerfile.prod -t swarmcraft-prod .
	@echo "Starting production container..."
	# We need redis for prod too. Using docker compose for prod is often easier,
	# but here we'll just show a run command or reuse compose if we had a prod override.
	# For simplicity, let's assume we want to run just the API here, linking to existing redis if any.
	# BUT, since the user likely wants the full stack:
	# Let's use a temporary override file or just build the image and use compose with a prod config.
	# For now, simple docker run (assuming redis is managed separately or we should run it):
	docker run -d --restart unless-stopped --env-file .env -p 8000:8000 --name swarmcraft-prod swarmcraft-prod

redis:
	docker compose up redis -d

websocket:
	docker compose exec api uv run python demo/test_websocket.py

# Testing
test:
	pytest

test-integration:
	pytest -m integration

test-all:
	pytest -m "integration or not integration"

test-cov:
	pytest -m "integration or not integration" --cov=src/swarmcraft --cov-report=term-missing

# Housekeeping
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f .coverage
	rm -rf .pytest_cache
	rm -rf htmlcov