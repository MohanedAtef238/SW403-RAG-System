#!/bin/bash
set -e

echo "Building base Docker image (sw403-base)..."
docker build -f Dockerfile.base -t sw403-base .

echo "Starting all services..."
docker compose up -d --build

echo "Done. Use 'docker compose logs' to view logs."
