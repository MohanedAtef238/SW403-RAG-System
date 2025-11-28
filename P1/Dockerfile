FROM sw403-base

WORKDIR /app

# Copy project files
COPY src/ ./src/
COPY main.py ./

# Expose port (P1 uses 8001 by default)
EXPOSE 8001

# Environment variables
ENV PYTHONPATH=/app
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Command to run the application
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8001"]

# Copy project files
COPY src/ ./src/
COPY main.py ./

# Expose port (P1 uses 8001 by default)
EXPOSE 8001

# Environment variables
ENV PYTHONPATH=/app
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
     CMD curl -f http://localhost:8001/health || exit 1

# Command to run the application
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8001"]