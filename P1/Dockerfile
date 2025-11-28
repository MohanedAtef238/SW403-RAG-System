FROM sw403-base

WORKDIR /app

# Copy only P1 application code
COPY src/ ./src/
COPY main.py ./main.py

EXPOSE 8001

ENV PYTHONPATH=/app

CMD ["uv", "run", "python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8001"]