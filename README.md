# SW403 RAG System - Prototype Comparison: AST vs. Text Chunking

A comprehensive RAG system for **codebase analysis**, enabling direct comparison between **P1 (Regex-based Text Splitting)** and **P2 (AST-based Semantic Chunking)** using Qdrant vector storage and detailed performance metrics.

---

## Quick Start

### 1. Preparation

* **Place files to ingest/test** in the local `./shared_data` directory (required for Docker containers to access them).
* **Ensure ports 6333, 8001, and 8002 are free** (required for Qdrant and the two prototype API services).

### 2. Configuration (Optional)

Copy and customize the environment configuration:

```bash
cp .env.example .env
```

Key configurable parameters:
- `EMBEDDING_MODEL`: Choose embedding model (MiniLM, MPNet, CodeBERT, BGE)
- `EMBEDDING_BATCH_SIZE`: Tune for speed vs memory (8-128)
- `DEFAULT_TOP_K`: Number of results to return (1-20)
- `MEMORY_LIMIT`: Docker container memory limit (1g-16g)

See [Configuration](#configuration) section below for details.

### 3. Build & Start

**Recommended (PowerShell):**
```powershell
.\build.ps1
```

**Or manually:**
```bash
# First time or when dependencies change
docker build -t sw403-base .

# Build and start all services
docker compose up -d
```

**Services will be available at:**
- Qdrant: `http://localhost:6333`
- P1 API: `http://localhost:8001`
- P2 API: `http://localhost:8002`

---

## Prototypes Overview

The system implements two distinct chunking approaches, each maintaining its own isolated Qdrant collection (`functions_p1`, `functions_p2`).

| Prototype | Chunking Mechanism | Metadata Richness | Endpoint Port | Qdrant Collection |
| :--- | :--- | :--- | :--- | :--- |
| **P1** | Regex-based Text Splitting (Baseline) | Minimal | `8001` | `functions_p1` |
| **P2** | Python AST-based Parsing (Advanced) | Rich (semantic context) | `8002` | `functions_p2` |

-----

## API Usage

All API calls are made to the running Docker containers via their exposed ports.

| Prototype | Base URL |
| :--- | :--- |
| P1 | `http://localhost:8001` |
| P2 | `http://localhost:8002` |

### Ingest Example (PowerShell)

```powershell
Invoke-WebRequest -Uri "http://localhost:8001/ingest" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "file_paths": ["/app/shared_data/test_sample.py"],
    "recreate_collection": false
  }'
```

### Query Example (PowerShell)

```powershell
Invoke-WebRequest -Uri "http://localhost:8001/query" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "query": "Which function has a decorator",
    "top_k": 2,
    "similarity_threshold": 0.0
  }'
```

> **Note on `recreate_collection`:** Setting this to `true` will drop and rebuild the vector store collection, clearing all previous data. If `false`, new data is added to the existing collection.

### API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/health` | System health check |
| **POST** | `/query` | Search function embeddings with detailed results |
| **POST** | `/ingest` | Ingest Python files into vector store |
| **GET** | `/collection/info` | Collection info and vector count |
| **GET** | `/collection/functions` | List all functions in collection |

-----

## Configuration

The system is highly configurable through environment variables in `.env` file.

### Quick Configuration

```bash
cp .env.example .env
# Edit .env with your settings
docker compose down
docker compose up -d
```

### Embedding Model Experiments

Test different embedding models by changing `EMBEDDING_MODEL` in `.env`:

| Model | Dimensions | Size | Speed | Quality | Use Case |
|-------|------------|------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | 80MB | ⚡⚡⚡ | ⭐⭐⭐ | **Default**, balanced |
| `all-mpnet-base-v2` | 768 | 420MB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Best quality |
| `BAAI/bge-small-en-v1.5` | 384 | 130MB | ⚡⚡⚡ | ⭐⭐⭐⭐ | State-of-the-art |
| `microsoft/codebert-base` | 768 | 500MB | ⚡⚡ | ⭐⭐⭐⭐ | Code-specific |

**Example:**
```bash
# .env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

### Performance Tuning

**Batch Size** (speed vs memory):
```bash
EMBEDDING_BATCH_SIZE=64  # Faster, more memory
EMBEDDING_BATCH_SIZE=16  # Slower, less memory
```

**Memory Limits** (per container):
```bash
MEMORY_LIMIT=2g   # Low memory systems
MEMORY_LIMIT=8g   # High memory systems
```

**GPU Acceleration** (requires NVIDIA Docker):
```bash
USE_GPU=true
```

### Search Configuration

```bash
DEFAULT_TOP_K=10                      # More results per query
DEFAULT_SIMILARITY_THRESHOLD=0.5      # Only high-confidence results
```

### Configuration Parameter Reference

| Parameter | Default | Options | Impact |
|-----------|---------|---------|--------|
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | See table above | Quality vs Speed |
| `EMBEDDING_BATCH_SIZE` | 32 | 8-128 | Speed vs Memory |
| `EMBEDDING_MAX_TOKENS` | 400 | 128-512 | Context vs Speed |
| `DEFAULT_TOP_K` | 5 | 1-20 | Recall vs Precision |
| `DEFAULT_SIMILARITY_THRESHOLD` | 0.0 | 0.0-1.0 | Precision vs Recall |
| `MEMORY_LIMIT` | 4g | 1g-16g | Container limit |
| `USE_GPU` | false | true/false | Speed (if GPU available) |

-----

## Development

### Rebuild Workflow

**After code changes (fast, seconds):**
```powershell
.\build.ps1
```

**After dependency changes (slow, 5-10 minutes):**
```bash
docker build -t sw403-base --no-cache .  # Rebuild base with new dependencies
docker compose build
docker compose up -d
```

> **Note:** The base image (`sw403-base`) contains all shared dependencies (PyTorch, sentence-transformers, etc.) and is cached separately from P1/P2 application code for fast rebuilds.

**Check services:**
```bash
docker compose ps
docker compose logs -f p1
```

### Local Development (without Docker)

Use this workflow if you prefer to run the RAG services locally.

**Install Dependencies:**
```bash
uv sync
```

**Start Qdrant (Required):**
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

**Start Development Server:**
```bash
uv run python main.py serve
```

-----

## Evaluation Workflow

For comprehensive evaluation and comparison of P1 vs P2:

**1. Ingest data into both prototypes:**
```powershell
# See EVALUATION_GUIDE.md for complete ingestion commands
Invoke-WebRequest -Uri "http://localhost:8001/ingest" -Method POST -ContentType "application/json" -Body '{...}'
Invoke-WebRequest -Uri "http://localhost:8002/ingest" -Method POST -ContentType "application/json" -Body '{...}'
```

**2. Run evaluation:**
```bash
uv run python evaluation/runner.py
```

**3. Generate error analysis:**
```bash
uv run python evaluation/analyzer.py
```

**4. Create visualizations:**
```bash
uv run python evaluation/visualize.py
```

**Results will be saved in `results/` directory.** See `EVALUATION_GUIDE.md` for detailed instructions.

-----

## Performance and Limitations

### Performance Metrics

Each query returns comprehensive metrics for quantitative evaluation:

```json
{
  "metrics": {
    "request_latency_seconds": 0.045,
    "embedding_generation_time_seconds": 0.012,
    "qdrant_retrieval_time_seconds": 0.028,
    "model_memory_usage_mb": 156.7,
    "collection_size": 1247,
    "results_returned": 5
  }
}
```

### Expected Performance

**P1 (Regex-based):**
  * Excels at simple lookup queries (e.g., "Find the `User` class", "What does the `save_data` function do?")
  * Struggles with local context queries, cross-function dependencies, and complex relational queries
  * Fast ingestion, minimal metadata overhead

**P2 (AST-based):**
  * Better at understanding semantic context and code structure
  * Captures rich metadata (decorators, parameters, return types, docstrings)
  * Slower ingestion due to AST parsing, but more accurate retrieval for complex queries

### Model Caching

The embedding model is cached in memory at startup to eliminate cold-start delays. This is essential for realistic performance evaluation.

-----

## Troubleshooting

**Services not starting:**
```bash
docker compose ps              # Check container status
docker compose logs -f p1      # View P1 logs
docker compose logs -f p2      # View P2 logs
```

**Port conflicts:**
- Ensure ports 6333, 8001, 8002 are free
- Check with: `netstat -an | findstr "6333 8001 8002"`

**Collection errors:**
```powershell
# Reset collections by recreating them
Invoke-WebRequest -Uri "http://localhost:8001/ingest" -Method POST -ContentType "application/json" -Body '{"file_paths": [...], "recreate_collection": true}'
```

**Slow downloads/builds:**
- First build downloads large models (PyTorch, embeddings), takes 5-10 minutes
- Subsequent builds reuse cached base image, take seconds
- Use `.\build.ps1` for optimized workflow

**Out of memory:**
- Reduce `EMBEDDING_BATCH_SIZE` in `.env`
- Reduce `MEMORY_LIMIT` or use smaller model (MiniLM)

-----