FROM sw403-base

WORKDIR /app

COPY P3/src/ ./src/
COPY main.py ./main.py

EXPOSE 8003

ENV PYTHONPATH=/app
ENV base_server="https://api.groq.com/openai/v1"

CMD ["uv", "run", "python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8003"]