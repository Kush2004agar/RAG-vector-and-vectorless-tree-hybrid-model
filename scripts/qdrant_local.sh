#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="docker-compose.qdrant.yml"

case "${1:-}" in
  up)
    docker compose -f "$COMPOSE_FILE" up -d
    echo "Qdrant started at http://localhost:6333"
    ;;
  down)
    docker compose -f "$COMPOSE_FILE" down
    ;;
  logs)
    docker compose -f "$COMPOSE_FILE" logs -f qdrant
    ;;
  status)
    docker compose -f "$COMPOSE_FILE" ps
    ;;
  *)
    echo "Usage: $0 {up|down|logs|status}"
    exit 1
    ;;
esac
