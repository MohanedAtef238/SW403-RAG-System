FROM sw403-base

WORKDIR /app

COPY src/ ./src/
COPY main.py ./main.py

EXPOSE 8003

ENV PYTHONPATH=/app
ENV base_server="https://api.groq.com/openai/v1"
ENV OPENAI_API_KEY=dummy

RUN uv add langchain-qdrant langchain langchain-openai

CMD ["uv", "run", "python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8003"]