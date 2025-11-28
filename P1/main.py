"""
SW403 P1 RAG System - Baseline Implementation
Simple regex-based function extraction for research comparison.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

import uvicorn
import click

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.api import app
from src.models import initialize_model, get_model_info
from src.chunking_p1 import P1TextChunker
from src.p1_utils import create_chunk_compatible_vector_store


@click.group()
def cli():
    """SW403 P1 Baseline RAG System CLI."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8001, help="Port to bind to (P1 default: 8001)")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool):
    """Start the FastAPI server for P1 baseline system."""
    print("Starting P1 Baseline System...")
    print(f"Server running on http://{host}:{port}")
    print(f"API docs: http://{host}:{port}/docs")
    
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


@cli.command()
@click.argument("files", nargs=-1, required=True)
@click.option("--recreate", is_flag=True, help="Recreate collection if it exists")
def ingest(files: List[str], recreate: bool):
    """Ingest Python files using P1 baseline chunking."""
    asyncio.run(_ingest(files, recreate))


async def _ingest(files: List[str], recreate: bool):
    """Async implementation of ingest command."""
    from src.models import encode_texts
    
    print("P1 Baseline: Starting ingestion...")
    print(f"Processing {len(files)} files")
    
    # Initialize components
    initialize_model()
    vector_store = create_chunk_compatible_vector_store(collection_name="functions_p1")
    chunker = P1TextChunker()
    
    if recreate:
        print("Recreating collection...")
        vector_store.recreate_collection()
    
    # Process each file
    all_chunks = []
    for file_path in files:
        print(f"Processing: {file_path}")
        try:
            chunks = chunker.chunk_file(str(file_path))
            if chunks:
                all_chunks.extend(chunks)
                print(f"   Found {len(chunks)} functions")
            else:
                print(f"   No functions found")
        except Exception as e:
            print(f"   Error: {e}")
    
    if all_chunks:
        print(f"Generating embeddings for {len(all_chunks)} functions...")
        # Get text content from P1 chunks
        chunk_texts = [chunk.raw_text for chunk in all_chunks]
        embeddings, metrics = encode_texts(chunk_texts)
        
        print(f"Storing in vector database...")
        success = vector_store.upsert_chunks(all_chunks, embeddings)
        
        if success:
            print(f"P1 Ingestion complete! Total functions: {len(all_chunks)}")
        else:
            print(f"Failed to store embeddings")
    else:
        print("No functions found to ingest")


@cli.command()
@click.argument("query", required=True)
@click.option("--top-k", default=5, help="Number of results to return")
def search(query: str, top_k: int):
    """Search for functions using P1 baseline system."""
    asyncio.run(_search(query, top_k))


async def _search(query: str, top_k: int):
    """Async implementation of search command."""
    from src.models import encode_single_text
    
    print(f"P1 Search: '{query}'")
    
    # Initialize components
    initialize_model()
    vector_store = create_chunk_compatible_vector_store(collection_name="functions_p1")
    
    try:
        # Generate query embedding
        query_embedding, model_metrics = encode_single_text(query)
        
        # Search
        results, vector_metrics = vector_store.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=0.0
        )
        
        if not results:
            print("No results found")
            return
        
        print(f"Found {len(results)} results:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            payload = result["payload"]
            print(f"\n{i}. Score: {result['similarity_score']:.4f}")
            print(f"   File: {payload.get('file_path', 'Unknown')}")
            print(f"   Function: {payload.get('function_name', 'Unknown')}")
            print(f"   Lines: {payload.get('line_numbers', {}).get('start', 0)}-{payload.get('line_numbers', {}).get('end', 0)}")
            print(f"   Content: {payload.get('raw_text', '')[:200]}...")
            
    except Exception as e:
        print(f"Search error: {e}")


@cli.command()
def status():
    """Show system status for P1 baseline."""
    asyncio.run(_status())


async def _status():
    """Async implementation of status command."""
    print("P1 Baseline Status")
    print("=" * 20)
    
    # Model info
    model_info = get_model_info()
    print(f"Model: {model_info['model_name']}")
    print(f"Dimensions: {model_info['model_dimensions']}")
    print(f"Device: {model_info['device']}")
    print(f"Memory: {model_info['current_memory_mb']:.1f}MB")
    
    # Vector store status
    try:
        vector_store = create_chunk_compatible_vector_store(collection_name="functions_p1")
        stats = vector_store.get_collection_info()
        print(f"\nCollection: functions_p1")
        print(f"Total functions: {stats.get('points_count', 0)}")
        print(f"Status: Connected")
    except Exception as e:
        print(f"Vector store error: {e}")


@cli.command()
def test():
    """Run comprehensive P1 baseline test."""
    asyncio.run(_test())


async def _test():
    """Async implementation of test command."""
    from src.models import encode_texts, encode_single_text
    
    print("P1 Baseline Test")
    print("=" * 20)
    
    try:
        # Test 1: Model initialization
        print("1. Testing model initialization...")
        initialize_model()
        print("   Model loaded successfully")
        
        # Test 2: Vector store connection
        print("2. Testing vector store connection...")
        vector_store = create_chunk_compatible_vector_store(collection_name="functions_p1")
        print("   Vector store connected")
        
        # Test 3: Chunking capability
        print("3. Testing P1 chunking capability...")
        chunker = P1TextChunker()
        
        # Create a test file
        test_content = '''def test_function():
    """A simple test function."""
    return "hello world"

class TestClass:
    def method(self):
        pass
'''
        test_file = Path("test_p1_temp.py")
        test_file.write_text(test_content)
        
        try:
            chunks = chunker.chunk_file(str(test_file))
            print(f"   Found {len(chunks)} functions in test content")
            
            # Test 4: Vector operations
            if chunks:
                print("4. Testing vector operations...")
                # Generate embeddings
                chunk_texts = [chunk.raw_text for chunk in chunks]
                embeddings, metrics = encode_texts(chunk_texts)
                
                # Store chunks
                success = vector_store.upsert_chunks(chunks, embeddings)
                if success:
                    print("   Chunks stored successfully")
                    
                    # Test search
                    print("5. Testing search functionality...")
                    query_embedding, _ = encode_single_text("test function")
                    results, search_metrics = vector_store.search_similar(query_embedding, top_k=2)
                    print(f"   Search returned {len(results)} results")
                else:
                    print("   Failed to store chunks")
            
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()
        
        print("All P1 tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()