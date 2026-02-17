#!/bin/sh
# wait for Postgres
echo "Waiting for Postgres..."
# print the full host and port
echo "Postgres host: postgres"
while ! nc -z postgres 5432; do
  sleep 1
done

echo "Postgres is up - executing command"
echo "Command: $@"
exec "$@"
