
# SW403 RAG System - Prototype Comparison: AST vs. Text Chunking

---

## Quick Start

### Preparation

1. **Place files to ingest/test in `shared_data/`** (required for Docker containers to access them).
2. **Ensure ports 6333, 6334, 8001–8010 are free** (required for Qdrant and API services).

### Build

```bash
docker build -t sw403-base .
```

### Start Services

```bash
docker compose up --build
```

---

## Prototypes Overview

| Prototype | Chunking Mechanism         | Metadata Richness | Endpoint Port |
|-----------|---------------------------|-------------------|---------------|

# SW403 RAG System - Prototype Comparison: AST vs. Text Chunking

A comprehensive RAG system for **codebase analysis**, enabling direct comparison between **P1 (Regex-based Text Splitting)** and **P2 (AST-based Semantic Chunking)** using Qdrant vector storage and detailed performance metrics.

---

## Quick Start: Docker-Based Workflow

All prototypes run in separate Docker containers inheriting from a shared base image for flexible switching and isolated development.

### 1. Preparation

* **Place files to ingest/test** in the local `./shared_data` directory (required for Docker containers to access them).
* **Ensure ports 6333, 8001, and 8002 are free** (required for Qdrant and the two prototype API services).

### 2. Build the Base Image

Build the common base image that both P1 and P2 containers will inherit from.

```bash
docker build -t sw403-base .
```

### 3. Start All Services

This command builds and starts Qdrant, Prototype P1, and Prototype P2 services.

```bash
docker compose up --build
```

-----

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

### Ingest Example (P1)

This ingests files from the mounted `shared_data` volume into the P1 vector store.

```bash
curl -X POST "http://localhost:8001/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["/app/shared_data/test_sample.py"],
    "recreate_collection": false
  }'
```

### Query Example (P1)

This searches the P1 vector store. To query **P2**, change the port to `8002`.

```bash
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Which function has a decorator",
    "top_k": 2,
    "similarity_threshold": 0.0
  }'
```

> **Note on `recreate_collection`:** Setting this to `true` will drop and rebuild the vector store collection, clearing all previous data. If `false`, new data is added to the existing collection.

-----


## Standard API Endpoints

The following endpoints are available on each prototype container (`http://localhost:8001/...` or `http://localhost:8002/...`).

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/` | System status and health check. |
| **POST** | `/query` | Search function embeddings with detailed results. |
| **POST** | `/ingest` | Ingest Python files into vector store. |
| **GET** | `/status` | Detailed system and collection status. |

### CLI Integration Endpoints

These endpoints mirror CLI commands for Docker integration, allowing prototype selection via the request body.

#### POST `/query`

```json
{
  "query": "model initialization",
  "top_k": 5,
  "similarity_threshold": 0.0
}
```

-----

## Development

Use this workflow if you prefer to run the RAG services locally (outside of Docker Compose).

### Install Dependencies

```bash
uv sync
```

### Start Qdrant (Required Local Dependency)

If not using Docker Compose, manually start the Qdrant database:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### Start Development Server

```bash
uv run python main.py serve
```

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

  * **P1** excels at simple lookup queries (e.g., "Find the `User` class", "What does the `save_data` function do?").
  * **P1** is expected to struggle with local context queries, cross-function dependencies, and complex relational queries. These limitations justify the advanced techniques in P2.

### Model Caching

The embedding model is cached in memory at startup to eliminate cold-start delays. This is essential for realistic performance evaluation.

-----