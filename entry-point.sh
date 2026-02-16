#!/bin/sh
# wait for Postgres
echo "Waiting for Postgres..."
while ! nc -z $POSTGRES_HOST 5432; do
  sleep 1
done

echo "Running migrations..."
alembic upgrade head

echo "Starting FastAPI..."
uvicorn src.main:app --host 0.0.0.0 --port 7000
