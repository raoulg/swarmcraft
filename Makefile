# Makefile for SwarmCraft

.PHONY: help setup up down build redis test test-integration test-all test-cov clean

help:
	@echo "Available commands:"
	@echo "  setup            - Create .env and install dependencies (local dev)."
	@echo "  up               - Start services with Docker Compose."
	@echo "  down             - Stop and remove all services."
	@echo "  build            - Rebuild the Docker images."
	@echo "  redis            - Start only the Redis container in the background."
	@echo "  test             - Run tests."
	@echo "  clean            - Remove temporary files."

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		KEY=$$(openssl rand -hex 32 2>/dev/null || echo "fallback_key_$$(date +%s)"); \
		sed -i.bak "s|SWARM_API_KEY=.*|SWARM_API_KEY=$$KEY|" .env; \
		PUBLIC_IP=$$(curl -s -4 ifconfig.me || curl -s -4 icanhazip.com); \
		if [ -n "$$PUBLIC_IP" ]; then \
			echo "Detected Public IP: $$PUBLIC_IP"; \
			sed -i.bak "s|CORS_ORIGINS=.*|CORS_ORIGINS=http://localhost:5173,http://$$PUBLIC_IP|" .env; \
		fi; \
		rm -f .env.bak; \
		echo "Created .env with a unique SWARM_API_KEY and CORS origins"; \
	fi
	uv sync

up:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		KEY=$$(openssl rand -hex 32 2>/dev/null || echo "fallback_key_$$(date +%s)"); \
		sed -i.bak "s|SWARM_API_KEY=.*|SWARM_API_KEY=$$KEY|" .env; \
		PUBLIC_IP=$$(curl -s -4 ifconfig.me || curl -s -4 icanhazip.com); \
		if [ -n "$$PUBLIC_IP" ]; then \
			echo "Detected Public IP: $$PUBLIC_IP"; \
			sed -i.bak "s|CORS_ORIGINS=.*|CORS_ORIGINS=http://localhost:5173,http://$$PUBLIC_IP|" .env; \
		fi; \
		rm -f .env.bak; \
		echo "Created .env with a unique SWARM_API_KEY and CORS origins"; \
	fi
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

redis:
	docker compose up redis -d

test:
	pytest

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f .coverage
	rm -rf .pytest_cache
