#!/bin/sh
# wait for Postgres
echo "Waiting for Postgres..."
# print the full host and port
echo "Postgres host: $POSTGRES_HOST"
while ! nc -z $POSTGRES_HOST 5432; do
  sleep 1
done

echo "Running migrations..."
uv run alembic upgrade head

echo "Starting FastAPI..."
uv run uvicorn src.main:app --host 0.0.0.0 --port 7000
