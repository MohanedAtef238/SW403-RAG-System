"""
FastAPI application with comprehensive performance metrics.
Provides /ingest and /query endpoints for the P1 RAG system.
"""

import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .models import initialize_model, encode_texts, encode_single_text, get_model_info
from .chunking import create_chunker, FunctionChunk, CodeChunker
from .vector_store import create_vector_store, QdrantVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SW403 P1 Baseline RAG System",
    description="P1 Baseline: Simple text-based function extraction for research comparison",
    version="1.0.0-p1"
)

# Add CORS middleware for n8n integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
chunker: Optional[CodeChunker] = None
vector_store: Optional[QdrantVectorStore] = None


# CLI wrapper request/response models
class CLISearchRequest(BaseModel):
    """Request schema for CLI search functionality."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(3, description="Number of results to return", ge=1, le=20)


class CLISearchResult(BaseModel):
    """Individual search result."""
    function_name: str
    file_path: str
    line_range: str
    similarity: float
    code_snippet: str


class CLISearchResponse(BaseModel):
    """Response schema for CLI search."""
    query: str
    results: List[CLISearchResult]
    search_time: float
    memory_usage: str
    total_results: int


class CLIIngestRequest(BaseModel):
    """Request schema for CLI ingest functionality."""
    file_patterns: List[str] = Field(..., description="List of file paths or glob patterns")
    recreate: bool = Field(False, description="Whether to recreate the collection")


class CLIIngestResponse(BaseModel):
    """Response schema for CLI ingest."""
    success: bool
    processed_files: List[str]
    total_functions: int
    embedding_time: float
    memory_usage: str
    message: str


class CLITestResponse(BaseModel):
    """Response schema for CLI test functionality."""
    success: bool
    model_info: Dict[str, Any]
    collection_size: int
    sample_search: Dict[str, Any]
    performance_metrics: Dict[str, Any]  # Changed from Dict[str, float] to Dict[str, Any]
    message: str


# Pydantic models for request/response schemas
class IngestRequest(BaseModel):
    """Request schema for code ingestion."""
    file_paths: List[str] = Field(..., description="List of Python file paths to ingest")
    source_code: Optional[str] = Field(None, description="Optional source code string (if not reading from files)")
    recreate_collection: bool = Field(False, description="Whether to recreate the collection from scratch")


class IngestResponse(BaseModel):
    """Response schema for code ingestion."""
    success: bool
    message: str
    functions_processed: int
    processing_time: float
    embedding_time: float
    storage_time: float
    model_memory_mb: float


class QueryRequest(BaseModel):
    """Request schema for RAG queries."""
    query: str = Field(..., description="The search query")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    similarity_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    file_filter: Optional[str] = Field(None, description="Filter results by file path")
    function_name_filter: Optional[str] = Field(None, description="Filter results by function name")


class FunctionResult(BaseModel):
    """Individual function result."""
    function_name: str
    function_signature: str
    file_path: str
    relative_path: str
    line_numbers: Dict[str, int]
    original_chunk_text: str
    docstring: Optional[str]
    similarity_score: float
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    """Response schema for RAG queries."""
    success: bool
    query: str
    results: List[FunctionResult]
    total_results: int
    metrics: Dict[str, Any]


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_info: Dict[str, Any]
    vector_store_info: Dict[str, Any]


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup."""
    global chunker, vector_store
    
    logger.info("Initializing P1 RAG system...")
    
    try:
        # Initialize embedding model (this caches it in memory)
        logger.info("Loading embedding model...")
        initialize_model()
        
        # Initialize chunker
        logger.info("Initializing code chunker...")
        chunker = create_chunker()
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = create_vector_store()
        
        logger.info("P1 RAG system initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize P1 RAG system: {e}")
        raise


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system information."""
    return HealthResponse(
        status="healthy",
        model_info=get_model_info(),
        vector_store_info=vector_store.get_collection_info() if vector_store else {}
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_code(request: IngestRequest):
    """
    Ingest Python code files into the vector database.
    """
    start_time = time.time()
    
    try:
        if request.recreate_collection and vector_store:
            logger.info("Recreating collection as requested")
            vector_store.recreate_collection()
        
        # Extract function chunks
        logger.info(f"Processing {len(request.file_paths)} files...")
        
        if request.source_code and chunker:
            # Process inline source code
            chunks = chunker.chunk_file("inline_code.py", request.source_code)
        else:
            # Process files
            all_chunks = []
            for file_path in request.file_paths:
                if not Path(file_path).exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                if chunker:
                    file_chunks = chunker.chunk_file(file_path)
                else:
                    file_chunks = []
                all_chunks.extend(file_chunks)
            
            chunks = all_chunks
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No functions found in the provided code")
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} functions...")
        chunk_texts = [chunk.original_chunk_text for chunk in chunks]
        embeddings, model_metrics = encode_texts(chunk_texts)
        
        # Store in vector database
        logger.info("Storing embeddings in vector database...")
        storage_start = time.time()
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
        success = vector_store.upsert_chunks(chunks, embeddings)
        storage_time = time.time() - storage_start
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store embeddings")
        
        total_time = time.time() - start_time
        
        return IngestResponse(
            success=True,
            message=f"Successfully processed {len(chunks)} functions from {len(request.file_paths)} files",
            functions_processed=len(chunks),
            processing_time=total_time,
            embedding_time=model_metrics.embedding_time,
            storage_time=storage_time,
            model_memory_mb=model_metrics.memory_usage
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_code(request: QueryRequest):
    """
    Query the RAG system for similar code functions.
    """
    request_start_time = time.time()
    
    try:
        # Generate query embedding
        logger.info(f"Processing query: '{request.query}'")
        query_embedding, model_metrics = encode_single_text(request.query)
        
        # Search vector database
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
        search_results, vector_metrics = vector_store.search_similar(
            query_embedding=query_embedding,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            file_filter=request.file_filter,
            function_name_filter=request.function_name_filter
        )
        
        # Format results
        formatted_results = []
        for result in search_results:
            payload = result["payload"]
            function_result = FunctionResult(
                function_name=payload["function_name"],
                function_signature=payload["function_signature"],
                file_path=payload["file_path"],
                relative_path=payload["relative_path"],
                line_numbers=payload["line_numbers"],
                original_chunk_text=payload["original_chunk_text"],
                docstring=payload.get("metadata", {}).get("docstring"),
                similarity_score=result["similarity_score"],
                metadata=payload.get("metadata", {})
            )
            formatted_results.append(function_result)
        
        # Calculate total request time
        total_request_time = time.time() - request_start_time
        
        # Compile comprehensive metrics
        metrics = {
            "request_latency_seconds": total_request_time,
            "embedding_generation_time_seconds": model_metrics.embedding_time,
            "qdrant_retrieval_time_seconds": vector_metrics.retrieval_time,
            "model_memory_usage_mb": model_metrics.memory_usage,
            "collection_size": vector_metrics.collection_size,
            "search_parameters": vector_metrics.search_params,
            "query_length": len(request.query),
            "results_returned": len(formatted_results)
        }
        
        # Log detailed metrics
        logger.info(
            f"Query completed: {len(formatted_results)} results in {total_request_time:.3f}s "
            f"(embedding: {model_metrics.embedding_time:.3f}s, "
            f"search: {vector_metrics.retrieval_time:.3f}s, "
            f"memory: {model_metrics.memory_usage:.1f}MB)"
        )
        
        return QueryResponse(
            success=True,
            query=request.query,
            results=formatted_results,
            total_results=len(formatted_results),
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/collection/info")
async def get_collection_info():
    """Get information about the vector collection."""
    try:
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        return vector_store.get_collection_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")


@app.get("/collection/functions")
async def list_all_functions(limit: int = 100):
    """List all functions in the collection (for debugging)."""
    try:
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        functions = vector_store.get_all_functions(limit=limit)
        return {
            "total_functions": len(functions),
            "functions": functions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list functions: {str(e)}")


@app.post("/collection/recreate")
async def recreate_collection():
    """Recreate the vector collection from scratch."""
    try:
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        success = vector_store.recreate_collection()
        if success:
            return {"success": True, "message": "Collection recreated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to recreate collection")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to recreate collection: {str(e)}")


# Development utilities
@app.get("/debug/model-info")
async def debug_model_info():
    """Get detailed model information for debugging."""
    return get_model_info()


# CLI Wrapper Endpoints
@app.post("/cli/search", response_model=CLISearchResponse)
async def cli_search(request: CLISearchRequest):
    """CLI-style search endpoint with JSON response."""
    if not vector_store:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    start_time = time.time()
    
    try:
        # Encode query
        query_embedding, _ = encode_single_text(request.query)
        
        # Search
        results, metrics = vector_store.search_similar(
            query_embedding,
            top_k=request.top_k,
            similarity_threshold=0.0
        )
        
        # Format results
        search_results = []
        for result in results:
            payload = result["payload"]
            search_results.append(CLISearchResult(
                function_name=payload["function_name"],
                file_path=payload["file_path"],
                line_range=f"{payload['line_numbers']['start']}-{payload['line_numbers']['end']}",
                similarity=result["similarity_score"],
                code_snippet=payload["function_signature"][:200] + "..." if len(payload["function_signature"]) > 200 else payload["function_signature"]
            ))
        
        search_time = time.time() - start_time
        model_info = get_model_info()
        
        return CLISearchResponse(
            query=request.query,
            results=search_results,
            search_time=round(search_time, 3),
            memory_usage=f"{model_info['current_memory_mb']:.1f}MB",
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/cli/ingest", response_model=CLIIngestResponse)
async def cli_ingest(request: CLIIngestRequest):
    """CLI-style ingest endpoint with JSON response."""
    if not chunker or not vector_store:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        # Expand file patterns
        all_files = []
        for pattern in request.file_patterns:
            if "*" in pattern:
                import glob
                files = glob.glob(pattern)
                all_files.extend(files)
            else:
                all_files.append(pattern)
        
        # Filter existing files
        existing_files = []
        for file_path in all_files:
            path = Path(file_path)
            if path.exists():
                existing_files.append(str(path))
        
        if not existing_files:
            model_info = get_model_info()
            return CLIIngestResponse(
                success=False,
                processed_files=[],
                total_functions=0,
                embedding_time=0.0,
                memory_usage=f"{model_info['current_memory_mb']:.1f}MB",
                message="No valid files found"
            )
        
        # Recreate collection if requested
        if request.recreate:
            vector_store.recreate_collection()
        
        # Process files
        all_chunks = []
        processed_files = []
        
        for file_path in existing_files:
            try:
                chunks = chunker.chunk_file(file_path)
                all_chunks.extend(chunks)
                processed_files.append(file_path)
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue
        
        if not all_chunks:
            model_info = get_model_info()
            return CLIIngestResponse(
                success=False,
                processed_files=processed_files,
                total_functions=0,
                embedding_time=0.0,
                memory_usage=f"{model_info['current_memory_mb']:.1f}MB",
                message="No functions found to ingest"
            )
        
        # Generate embeddings
        start_time = time.time()
        chunk_texts = [chunk.original_chunk_text for chunk in all_chunks]
        embeddings, metrics = encode_texts(chunk_texts)
        embedding_time = time.time() - start_time
        
        # Store in vector database
        success = vector_store.upsert_chunks(all_chunks, embeddings)
        
        model_info = get_model_info()
        
        return CLIIngestResponse(
            success=success,
            processed_files=processed_files,
            total_functions=len(all_chunks),
            embedding_time=round(embedding_time, 3),
            memory_usage=f"{model_info['current_memory_mb']:.1f}MB",
            message=f"Successfully ingested {len(all_chunks)} functions from {len(processed_files)} files" if success else "Ingestion failed"
        )
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/cli/test", response_model=CLITestResponse)
async def cli_test():
    """CLI-style test endpoint with JSON response."""
    try:
        # Get model info
        model_info = get_model_info()
        
        # Get collection size
        collection_size = 0
        if vector_store:
            collection_size = vector_store.get_collection_size()
        
        # Perform sample search if data exists
        sample_search = {}
        if collection_size > 0 and vector_store:
            try:
                query_embedding, _ = encode_single_text("test function")
                results, metrics = vector_store.search_similar(
                    query_embedding,
                    top_k=1,
                    similarity_threshold=0.0
                )
                sample_search = {
                    "query": "test function",
                    "results_found": len(results),
                    "search_time": metrics.retrieval_time
                }
            except Exception as e:
                sample_search = {"error": str(e)}
        
        # Performance metrics
        performance_metrics = {
            "model_memory_mb": model_info["current_memory_mb"],
            "model_device": model_info["device"],
            "collection_size": collection_size
        }
        
        return CLITestResponse(
            success=True,
            model_info=model_info,
            collection_size=collection_size,
            sample_search=sample_search,
            performance_metrics=performance_metrics,
            message="System test completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Test error: {e}")
        raise HTTPException(status_code=500, detail=f"System test failed: {str(e)}")


if __name__ == "__main__":
    # For development/testing
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )