.PHONY: help build up down restart logs health check-env dev install clean backup restore

# Default target
help:
	@echo "Cortex Memory API - Available commands:"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make check-env    - Check required environment variables"
	@echo "  make build        - Build Docker images"
	@echo "  make up           - Start all services"
	@echo "  make health       - Check service health"
	@echo ""
	@echo "🔧 Development:"
	@echo "  make dev          - Start services for development"
	@echo "  make install      - Install Python dependencies"
	@echo "  make serve        - Run API server locally (requires deps services)"
	@echo ""
	@echo "📊 Management:"
	@echo "  make logs         - View service logs"
	@echo "  make restart      - Restart all services"
	@echo "  make down         - Stop all services"
	@echo "  make clean        - Stop services and remove volumes"
	@echo ""
	@echo "🩺 Health & Debug:"
	@echo "  make status       - Show running services"
	@echo "  make shell        - Access API container shell"
	@echo "  make db-shell     - Access PostgreSQL shell"
	@echo ""
	@echo "💾 Data Management:"
	@echo "  make backup       - Backup PostgreSQL and ChromaDB data"
	@echo "  make restore      - Restore from backups"
	@echo "  make backup-auto  - Set up automated daily backups"

# Environment check
check-env:
	@echo "🔍 Checking environment configuration..."
	@if [ ! -f .env ]; then \
		echo "❌ .env file not found. Copying from .env.example..."; \
		cp .env.example .env; \
		echo "✅ Created .env file. Please edit it with your configuration."; \
		exit 1; \
	fi
	@echo "✅ Environment configuration checked, please make sure that the values are properly set in the .env file"

# Build Docker images
build: check-env
	@echo "🔨 Building Cortex API Docker image..."
	cd docker && docker-compose build
	@echo "✅ Build complete"

# Start all services
up: check-env
	@echo "🚀 Starting Cortex Memory API services..."
	@echo "📁 Ensuring data directories exist for persistent volumes..."
	@mkdir -p docker/data/postgres docker/data/chroma
	cd docker && docker-compose up -d
	@echo "⏳ Waiting for services to start (this may take up to 60 seconds)..."
	@echo "   ChromaDB is initializing database..."
	@for i in $$(seq 1 12); do \
		sleep 5; \
		if curl -s --connect-timeout 2 http://localhost:7003/api/v1/heartbeat > /dev/null 2>&1; then \
			echo "   ChromaDB is ready!"; \
			break; \
		else \
			echo -n "."; \
		fi; \
	done; \
	echo ""
	@$(MAKE) health
	@echo ""
	@echo "🎉 Cortex Memory API is running!"
	@echo "   📚 API Docs: http://localhost:7001/docs"
	@echo "   🩺 Health: http://localhost:7001/api/v1/health"
	@echo "   💾 Data stored in: docker/data/"

# Stop all services
down:
	@echo "🛑 Stopping Cortex Memory API services..."
	cd docker && docker-compose down
	@echo "✅ Services stopped"

# Restart all services
restart:
	@echo "🔄 Restarting Cortex Memory API services..."
	cd docker && docker-compose restart
	@$(MAKE) health

# View logs
logs:
	@echo "📋 Service logs (Press Ctrl+C to exit):"
	cd docker && docker-compose logs -f

# Check service health
health:
	@echo "🩺 Checking service health..."
	@echo -n "   API Server: "
	@if curl -s http://localhost:7001/api/v1/health > /dev/null 2>&1; then \
		echo "✅ Healthy"; \
	else \
		echo "❌ Unhealthy"; \
	fi
	@echo -n "   ChromaDB: "
	@if curl -s --connect-timeout 5 http://localhost:7003/api/v1/heartbeat > /dev/null 2>&1; then \
		echo "✅ Healthy"; \
	else \
		echo "❌ Unhealthy (may still be starting up)"; \
	fi
	@echo -n "   PostgreSQL: "
	@if cd docker && docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then \
		echo "✅ Healthy"; \
	else \
		echo "❌ Unhealthy"; \
	fi

# Show running services status
status:
	@echo "📊 Service Status:"
	cd docker && docker-compose ps

# Development mode (with file watching)
dev: check-env
	@echo "🔧 Starting development environment..."
	cd docker && docker-compose -f docker-compose.yml up -d chromadb postgres
	@echo "⏳ Waiting for dependencies..."
	@sleep 5
	@echo "🚀 Starting API in development mode..."
	@echo "   Dependencies running, start API with: make serve"

# Install Python dependencies locally
install:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Run API server locally (requires dependencies)
serve: install
	@echo "🚀 Starting Cortex API server locally..."
	@echo "   Make sure dependencies are running: make dev"
	uvicorn app.main:app --reload --host 0.0.0.0 --port 7001

# Clean up everything (including volumes)
clean:
	@echo "🧹 Cleaning up Cortex Memory API..."
	cd docker && docker-compose down -v --remove-orphans
	@echo "🗑️  Removing unused Docker resources..."
	docker system prune -f
	@echo "✅ Cleanup complete"

# Access API container shell
shell:
	@echo "🐚 Accessing Cortex API container shell..."
	cd docker && docker-compose exec cortex-api /bin/bash

# Access PostgreSQL shell
db-shell:
	@echo "🐘 Accessing PostgreSQL shell..."
	cd docker && docker-compose exec postgres psql -U postgres -d cortex_api

# Quick test of API functionality
test-api: health
	@echo "🧪 Testing API functionality..."
	@echo -n "   Registration: "
	@if curl -s -X POST "http://localhost:7001/api/v1/auth/register" \
		-H "Content-Type: application/json" \
		-d '{"username":"test_$(shell date +%s)","email":"test@example.com","password":"test123"}' \
		| grep -q "api_key"; then \
		echo "✅ Working"; \
	else \
		echo "❌ Failed"; \
	fi

# Backup data
backup:
	@echo "💾 Creating backup of PostgreSQL and ChromaDB data..."
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@echo "📊 Backing up PostgreSQL database..."
	cd docker && docker-compose exec -T postgres pg_dump -U postgres cortex_api > ../backups/$(shell date +%Y%m%d_%H%M%S)/postgres_backup.sql
	@echo "🔍 Backing up ChromaDB data..."
	@cp -r docker/data/chroma backups/$(shell date +%Y%m%d_%H%M%S)/chroma_backup 2>/dev/null || echo "ChromaDB data directory not found"
	@echo "✅ Backup completed: backups/$(shell date +%Y%m%d_%H%M%S)/"

# Restore data from backup
restore:
	@echo "📥 Available backups:"
	@ls -la backups/ 2>/dev/null || echo "No backups found"
	@echo ""
	@echo "To restore, run: make restore-from BACKUP=YYYYMMDD_HHMMSS"

# Restore from specific backup
restore-from:
	@if [ -z "$(BACKUP)" ]; then echo "❌ Please specify BACKUP=YYYYMMDD_HHMMSS"; exit 1; fi
	@echo "📥 Restoring from backup: $(BACKUP)"
	@echo "⚠️  This will overwrite current data. Press Ctrl+C to cancel..."
	@sleep 5
	@echo "🛑 Stopping services..."
	@$(MAKE) down
	@echo "🗑️  Cleaning current data..."
	@rm -rf docker/data/postgres/* docker/data/chroma/*
	@echo "📊 Restoring PostgreSQL..."
	@$(MAKE) up-deps
	@sleep 10
	cd docker && docker-compose exec -T postgres psql -U postgres -d cortex_api < ../backups/$(BACKUP)/postgres_backup.sql
	@echo "🔍 Restoring ChromaDB..."
	@cp -r backups/$(BACKUP)/chroma_backup/* docker/data/chroma/ 2>/dev/null || echo "No ChromaDB backup found"
	@echo "✅ Restore completed"

# Start only dependencies (for restore process)
up-deps:
	@mkdir -p docker/data/postgres docker/data/chroma
	cd docker && docker-compose up -d postgres chromadb

# Set up automated daily backups
backup-auto:
	@echo "⏰ Setting up automated daily backups..."
	@(crontab -l 2>/dev/null; echo "0 2 * * * cd $(PWD) && make backup") | crontab -
	@echo "✅ Daily backup scheduled at 2 AM"
	@echo "   View with: crontab -l"
	@echo "   Remove with: crontab -e"

# Development workflow
dev-workflow: clean build up health
	@echo ""
	@echo "🎯 Development environment ready!"
	@echo "   Next steps:"
	@echo "   1. Edit your code in app/"
	@echo "   2. View logs: make logs"
	@echo "   3. Test API: make test-api"