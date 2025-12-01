FROM sw403-base

WORKDIR /app

COPY src/ ./src/
COPY main.py ./main.py

EXPOSE 8003

ENV PYTHONPATH=/app
ENV base_server="https://api.groq.com/openai/v1"
ENV OPENAI_API_KEY="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"

RUN [ "uv","run", "python","./P3/src/rag.py" ]

CMD ["uv", "run", "python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8003"]