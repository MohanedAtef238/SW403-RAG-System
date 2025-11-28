FROM sw403-base

WORKDIR /app

COPY src/ ./src/
COPY main.py ./main.py

EXPOSE 8002

ENV PYTHONPATH=/app

CMD ["uv", "run", "python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8002"]