# SW403 RAG System - Prototype Comparison (P1 vs P2)

A comprehensive RAG system implementing two distinct chunking approaches for codebase analysis. This system enables direct comparison between baseline text-based chunking (P1) and advanced AST-based semantic chunking (P2) using Qdrant vector storage and comprehensive performance metrics.

## Prototypes Overview

### P1 (Baseline) - Simple Text Splitting
- **Chunking**: Regex-based function detection with basic indentation parsing
- **Metadata**: Minimal (function names, line numbers, raw text)
- **Performance**: Fast ingestion, lower semantic quality
- **Use Case**: Baseline performance measurement

### P2 (Smarter Chunks) - AST-Based Semantic Chunking  
- **Chunking**: Full Python AST parsing with semantic boundaries
- **Metadata**: Rich (signatures, docstrings, complexity, decorators)
- **Performance**: Thorough analysis, higher semantic quality
- **Use Case**: Enhanced semantic understanding

## Features

### Dual Prototype Architecture

- **P1 & P2 Coexistence**: Both prototypes run independently with separate Qdrant collections
- **Direct Comparison**: Built-in comparison tools to evaluate chunking approaches
- **Flexible Switching**: Choose prototype via CLI `--prototype` flag
- **Performance Metrics**: Detailed timing and quality measurements for both approaches

### Core RAG Pipeline

- **384-dimensional embeddings** using `all-MiniLM-L6-v2` model
- **Docker Qdrant storage** with cosine similarity search  
- **Separate collections**: `functions_p1` and `functions_p2`
- **Comprehensive metadata** (varies by prototype)

### API Endpoints

#### Original Endpoints
- `POST /ingest` - Process Python files (uses P2 by default)
- `POST /query` - Semantic search for similar functions  
- `GET /health` - System health and configuration info (for debugging)
- `GET /collection/info` - Vector database statistics

#### New CLI-Wrapper Endpoints
- `POST /cli/search` - Search with JSON response and prototype selection
- `POST /cli/ingest` - Ingest with file patterns and prototype choice
- `GET /cli/test` - System health test with performance metrics (for debugging)

### Performance Monitoring

- **Prototype-specific metrics**: Compare P1 vs P2 performance
- **Request latency** tracking (total response time)
- **Embedding generation time** (isolated model performance)  
- **Qdrant retrieval time** (vector database efficiency)
- **Model memory usage** (resource consumption monitoring)
- **Similarity score analysis** (semantic quality comparison)

## Quick Start

### 1. Install Dependencies & Start Qdrant

```bash
# Install Python dependencies
uv sync

# Start Qdrant Docker container (required)
docker run -p 6333:6333 qdrant/qdrant
# Or restart existing container: docker start <container_name>
```

### 2. Compare Prototypes

```bash
# Compare P1 vs P2 chunking approaches
uv run python main.py compare src/models.py src/api.py
```

### 3. Ingest Data (Choose Prototype)

```bash
# Ingest with P1 (baseline text-based)
uv run python main.py ingest src/*.py --prototype P1 --recreate

# Ingest with P2 (AST-based semantic)
uv run python main.py ingest src/*.py --prototype P2 --recreate
```

### 4. Search & Compare Results

```bash
# Search P1 collection
uv run python main.py search "model initialization" --prototype P1 --top-k 3

# Search P2 collection  
uv run python main.py search "model initialization" --prototype P2 --top-k 3
```

### 5. Start API Server

```bash
# Start FastAPI server with all endpoints
uv run python main.py serve
# API docs: http://localhost:8000/docs
```

## API Usage Examples

### CLI-Style API Endpoints (New)

```bash
# Search via API with prototype selection
curl -X POST "http://localhost:8000/cli/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "model initialization", "top_k": 3}'

# Ingest via API with file patterns
curl -X POST "http://localhost:8000/cli/ingest" \
  -H "Content-Type: application/json" \
  -d '{"file_patterns": ["src/*.py"], "recreate": false}'

# System health test
curl -X GET "http://localhost:8000/cli/test"
```

### Original API Endpoints

```bash
# Original query endpoint (uses P2)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "FastAPI endpoint", "top_k": 5, "similarity_threshold": 0.1}'

# Original ingest endpoint (uses P2)
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"file_paths": ["src/models.py", "src/api.py"]}'
```

## CLI Commands

### Core Commands

```bash
# Server Management
uv run python main.py serve                    # Start FastAPI server

# Prototype Comparison (Most Important)
uv run python main.py compare FILE...          # Compare P1 vs P2 on files

# Data Ingestion
uv run python main.py ingest FILE... --prototype P1    # Ingest with baseline
uv run python main.py ingest FILE... --prototype P2    # Ingest with AST-based
uv run python main.py ingest FILE... --recreate        # Recreate collection

# Search & Query
uv run python main.py search QUERY --prototype P1      # Search P1 collection
uv run python main.py search QUERY --prototype P2      # Search P2 collection
uv run python main.py search QUERY --top-k 10          # Limit results
```

### Command Examples

```bash
# Compare chunking approaches
uv run python main.py compare src/models.py src/api.py

# Ingest with both prototypes
uv run python main.py ingest src/*.py --prototype P1 --recreate
uv run python main.py ingest src/*.py --prototype P2 --recreate  

# Compare search results
uv run python main.py search "model initialization" --prototype P1
uv run python main.py search "model initialization" --prototype P2
```

## The Compare Command - Core Research Tool

The `compare` command is **the most important feature** for research evaluation. It directly compares how P1 and P2 chunk the same files, revealing their fundamental differences.

### Why Use Compare?

1. **Validation**: Ensures both prototypes find the same functions (or explains differences)
2. **Quality Assessment**: Shows which approach misses functions or creates invalid chunks  
3. **Research Foundation**: Provides quantitative basis for performance claims
4. **Debugging**: Identifies chunking issues before ingestion

### Compare Output Example

```bash
uv run python main.py compare src/models.py
```

```text
=== Prototype Comparison: P1 vs P2 ===

P1: Baseline - Simple Text Splitting
   Uses regex patterns to find function definitions and simple indentation logic

P2: Smarter Chunks - AST-based
   Uses Python AST for semantic function extraction with rich metadata

Analyzing: src/models.py
--------------------------------------------------
P1 found: 8 functions
P2 found: 8 functions  
Agreement: 100.0%
```

### Interpreting Results

- **100% Agreement**: Both find identical functions (ideal)
- **Only P1 found**: P1's regex caught something P2's AST missed (investigate)
- **Only P2 found**: P2's semantic parsing found valid functions P1 missed (expected)
- **Low Agreement**: Fundamental chunking differences (needs investigation)

### Research Use Cases

1. **Baseline Validation**: Prove P1 captures obvious functions correctly
2. **P2 Superiority**: Show P2 finds functions P1 misses (complex signatures, decorators)
3. **Edge Case Discovery**: Find where simple regex fails vs AST parsing
4. **Consistency Check**: Ensure both prototypes work on your test corpus

## API Documentation

Interactive API docs available at: `http://localhost:8000/docs`

### Standard API Endpoints

**GET /** - System status and health check  
**POST /search** - Search function embeddings with detailed results  
**POST /ingest** - Ingest Python files into vector store  
**GET /status** - Detailed system and collection status  
**POST /test** - Run comprehensive system test  

### CLI Integration Endpoints

These endpoints mirror CLI commands for Docker integration:

#### POST /cli/search

```json
{
  "query": "model initialization",
  "top_k": 5,
  "prototype": "P1"
}
```

#### POST /cli/ingest

```json
{
  "files": ["src/models.py", "src/api.py"],
  "recreate_collection": true,
  "prototype": "P2"
}
```

#### POST /cli/test

```json
{
  "prototype": "P2"
}
```

## Architecture

The system maintains two separate Qdrant collections (`functions_p1` and `functions_p2`) to ensure complete prototype independence while supporting cross-compatibility for search and comparison operations.

## Development

### Quick Setup

```bash
# Install dependencies
uv sync

# Start Qdrant (if not using Docker)
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Start development server
uv run python main.py serve
```

### Testing Workflow

```bash
# Compare prototypes on test files
uv run python main.py compare src/*.py

# Ingest with both prototypes
uv run python main.py ingest src/*.py --prototype P1 --recreate
uv run python main.py ingest src/*.py --prototype P2 --recreate

# Test search across both
uv run python main.py search "function definition" --prototype P1
uv run python main.py search "function definition" --prototype P2
```

## Docker Integration

```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage
    
  sw403:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - qdrant
    environment:
      - QDRANT_HOST=qdrant
```

## Notes

- **Prototype Independence**: P1 and P2 use separate collections, no data mixing
- **Cross-Compatibility**: Search works across both prototypes despite different chunk formats  
- **Performance**: P1 focuses on speed, P2 on semantic accuracy
- **Research Ready**: Compare command provides quantitative evaluation framework
- **Docker Ready**: All functionality available via API for containerized environments
│   Chunking      │    │   Tracking       │    │    line numbers)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Performance Metrics

Each query returns comprehensive metrics:

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

## Expected Performance (Category 1 Queries)

P1 should excel at **simple lookup queries**:
- "Find the User class"
- "What does the save_data function do?"
- "Functions that handle authentication"

## Known Limitations (Categories 2 & 3)

P1 is expected to struggle with:
- "Local context queries" (variables defined outside functions)
- "Cross-function dependencies" (following logic across files)
- "Complex relational queries" (full authentication flow)

These limitations justify the advanced techniques in P2-P4.

## Development

### Project Structure
```
sw403/
├── main.py                 # CLI and server launcher
├── src/
│   ├── api.py              # FastAPI endpoints
│   ├── models.py           # Embedding model management
│   ├── chunking.py         # AST-based function extraction
│   └── vector_store.py     # Qdrant operations
├── qdrant_storage/         # Vector database files
└── pyproject.toml          # Dependencies and config
```

### Model Caching

The embedding model is cached in memory at startup to eliminate cold-start delays:
- **First request**: ~2-3 seconds (model loading)
- **Subsequent requests**: <100ms (cached model)

This is essential for n8n integration and realistic performance evaluation.
