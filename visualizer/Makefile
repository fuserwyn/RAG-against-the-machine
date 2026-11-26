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

## no Docker: same detached server as `make local` in the repository root
local:
	$(MAKE) -C .. local
