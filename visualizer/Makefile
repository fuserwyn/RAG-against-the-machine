.PHONY: up down logs rebuild ps local

up:
	docker compose up -d --build
	@echo "open http://localhost:8080"

down:
	docker compose down

logs:
	docker compose logs -f

rebuild:
	docker compose build --no-cache

ps:
	docker compose ps
	docker network inspect rag-net --format '{{range .Containers}}{{.Name}} {{.IPv4Address}}{{"\n"}}{{end}}'

## no Docker: run the trace server with the project's uv env and serve the UI on :8080
local:
	cd .. && PYTHONPATH=. uv run python visualizer/engine/trace_server.py \
		--host 127.0.0.1 --port 8080 --static visualizer/viz
