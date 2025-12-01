FROM python:3.12-slim

# Install uv (fast package installer)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy only dependency files first (for better caching)
COPY pyproject.toml .
COPY uv.lock .

# Install dependencies in a separate layer (cached unless dependencies change)
# Use --frozen to avoid re-locking and speed up builds
RUN uv sync --frozen --no-dev

# Note: P1/P2 containers will copy their own code
# This base image contains only shared dependencies
