FROM python:3.12-slim

# Install uv (fast package installer)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Install root dependencies
RUN uv sync --no-dev

# Copy full project (but not P1/P2 code)
COPY . .
