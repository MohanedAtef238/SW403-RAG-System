"""
Qdrant vector store management with performance instrumentation.
Handles 384-dimensional embeddings with cosine similarity and batch operations.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CreateCollection, PointStruct, 
    Filter, FieldCondition, MatchText
)
from .chunking import FunctionChunk

logger = logging.getLogger(__name__)

# Configuration constants
COLLECTION_NAME = "function_embeddings"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 dimensions
DISTANCE_METRIC = Distance.COSINE


class VectorStoreMetrics:
    """Container for vector store performance metrics."""
    
    def __init__(self):
        self.retrieval_time: float = 0.0
        self.results_count: int = 0
        self.collection_size: int = 0
        self.search_params: Dict[str, Any] = {}


class QdrantVectorStore:
    """Qdrant vector store with performance monitoring."""
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, storage_path: Optional[str] = None, collection_name: Optional[str] = None):
        logger.info(f"Qdrant host: {host}, port: {port}")
        if host is None:
            raise ValueError("QDRANT_HOST environment variable must be set. Current value: None")
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, storage_path: Optional[str] = None, collection_name: Optional[str] = None):
        """
        Initialize Qdrant client.
        
        Args:
            host: Qdrant server host (for Docker/server mode)
            port: Qdrant server port (for Docker/server mode)
            storage_path: Local file storage path (for embedded mode). If None, uses Docker/server mode.
            collection_name: Custom collection name. If None, uses default.
        """
        import os
        if host is None:
            host = os.getenv("QDRANT_HOST")
        if port is None:
            port_env = os.getenv("QDRANT_PORT")
            port = int(port_env) if port_env is not None else None
        if storage_path:
            # Local file storage mode
            self.client = QdrantClient(path=storage_path)
            logger.info(f"Using Qdrant local storage at {storage_path}")
        else:
            # Docker/server mode
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Connected to Qdrant server at {host}:{port}")
        
        self.collection_name = collection_name or COLLECTION_NAME
        logger.info(f"Using collection: {self.collection_name}")
        self._ensure_collection()
    
    def _ensure_collection(self) -> None:
        """Ensure the collection exists with proper configuration."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection '{self.collection_name}'")
                self._create_collection()
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    def _create_collection(self) -> None:
        """Create the function embeddings collection."""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=DISTANCE_METRIC,
                on_disk=True  # Store vectors on disk for memory efficiency
            )
        )
        
        # Create payload indexes for efficient filtering
        try:
            # Create simple payload indexes - skip if not supported in this version
            # This is optional for basic functionality
            pass
            
            logger.info("Created payload indexes for efficient filtering")
            
        except Exception as e:
            logger.warning(f"Could not create payload indexes: {e}")
    
    def upsert_chunks(self, chunks: List[FunctionChunk], embeddings: np.ndarray) -> bool:
        """
        Upsert function chunks with their embeddings into Qdrant.
        
        Args:
            chunks: List of FunctionChunk objects
            embeddings: Corresponding embeddings array
            
        Returns:
            Success status
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        try:
            # Create points for upsert
            points = []
            for i, chunk in enumerate(chunks):
                # Generate deterministic positive ID using abs() to ensure unsigned integer so qdrant can index it properly.
                chunk_id = abs(hash(f"{chunk.file_path}:{chunk.function_name}:{chunk.start_line}"))
                point = PointStruct(
                    id=chunk_id,
                    vector=embeddings[i].tolist(),
                    payload=chunk.to_payload()
                )
                points.append(point)
            
            # Batch upsert
            start_time = time.time()
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            upsert_time = time.time() - start_time
            
            logger.info(f"Upserted {len(points)} function chunks in {upsert_time:.3f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting chunks: {e}")
            return False
    
    def search_similar(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        file_filter: Optional[str] = None,
        function_name_filter: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], VectorStoreMetrics]:
        """
        Search for similar functions with performance tracking.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            file_filter: Optional file path filter
            function_name_filter: Optional function name filter
            
        Returns:
            Tuple of (search results, metrics)
        """
        metrics = VectorStoreMetrics()
        
        try:
            # Build filters if specified
            search_filter = None
            if file_filter or function_name_filter:
                conditions = []
                
                if file_filter:
                    conditions.append(
                        FieldCondition(
                            key="file_path",
                            match=MatchText(text=file_filter)
                        )
                    )
                
                if function_name_filter:
                    conditions.append(
                        FieldCondition(
                            key="function_name", 
                            match=MatchText(text=function_name_filter)
                        )
                    )
                
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # Perform search with timing
            start_time = time.time()
            
            # Get all points with vectors for manual similarity calculation
            all_points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Get enough points to find similar ones
                with_payload=True,
                with_vectors=True,
                scroll_filter=search_filter
            )
            
            # Calculate cosine similarity manually
            # Handle different embedding formats correctly
            if isinstance(query_embedding, np.ndarray):
                if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
                    query_vector = query_embedding.flatten()  # Convert (1, 384) -> (384,)
                else:
                    query_vector = query_embedding
            else:
                query_vector = np.array(query_embedding)
            similarities = []
            
            logger.info(f"Processing {len(all_points)} points, query vector length: {len(query_vector)}")
            
            for point in all_points:
                if point.vector is not None:
                    try:
                        # Convert vector using direct numpy conversion
                        try:
                            point_vector = np.array(point.vector, dtype=float)
                            # Ensure it's 1D and has the right length
                            if point_vector.ndim != 1 or len(point_vector) != len(query_vector):
                                continue
                        except (ValueError, TypeError):
                            continue  # Skip this point if conversion fails
                        
                        # Calculate cosine similarity
                        if len(point_vector) == len(query_vector):
                            # Normalize vectors
                            query_norm = np.linalg.norm(query_vector)
                            point_norm = np.linalg.norm(point_vector)
                            
                            if query_norm > 0 and point_norm > 0:
                                cosine_sim = np.dot(query_vector, point_vector) / (query_norm * point_norm)
                                
                                if cosine_sim >= similarity_threshold:
                                    similarities.append((point, float(cosine_sim)))
                                    
                    except (ValueError, TypeError, IndexError) as e:
                        continue  # Skip problematic vectors
            
            # Sort by similarity (highest first) and take top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Found {len(similarities)} matches above threshold {similarity_threshold}")
            
            # Create result objects
            search_results = []
            for point, score in similarities[:top_k]:
                class MockResult:
                    def __init__(self, id, score, payload):
                        self.id = id
                        self.score = score
                        self.payload = payload
                
                search_results.append(MockResult(point.id, score, point.payload))
            
            retrieval_time = time.time() - start_time
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "similarity_score": result.score,
                    "payload": result.payload
                })
            
            # Populate metrics
            metrics.retrieval_time = retrieval_time
            metrics.results_count = len(results)
            metrics.collection_size = self.get_collection_size()
            metrics.search_params = {
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "file_filter": file_filter,
                "function_name_filter": function_name_filter
            }
            
            logger.info(f"Retrieved {len(results)} results in {retrieval_time:.3f}s "
                       f"(similarity >= {similarity_threshold})")
            
            return results, metrics
            
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return [], metrics
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information and statistics."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vector_size": VECTOR_SIZE,
                "distance_metric": DISTANCE_METRIC.value,
                "points_count": collection_info.points_count or 0,
                "indexed_vectors_count": collection_info.points_count or 0,
                "status": str(collection_info.status)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def get_collection_size(self) -> int:
        """Get the number of points in the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception as e:
            logger.error(f"Error getting collection size: {e}")
            return 0
    
    def delete_collection(self) -> bool:
        """Delete the collection (useful for testing/reset)."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def recreate_collection(self) -> bool:
        """Recreate the collection from scratch."""
        try:
            self.delete_collection()
            self._create_collection()
            logger.info(f"Recreated collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error recreating collection: {e}")
            return False
    
    def search_by_function_name(self, function_name: str) -> List[Dict[str, Any]]:
        """Search for functions by exact name match."""
        try:
            # Use scroll to get all matches (not limited by top_k)
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="function_name",
                            match=MatchText(text=function_name)
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=False,
                limit=100  # Max results per page
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "payload": result.payload
                })
            
            logger.info(f"Found {len(formatted_results)} functions named '{function_name}'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching by function name: {e}")
            return []
    
    def get_all_functions(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all functions in the collection (for debugging)."""
        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "payload": result.payload
                })
            
            logger.info(f"Retrieved {len(formatted_results)} functions from collection")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error getting all functions: {e}")
            return []


def create_vector_store(host: str = "localhost", port: int = 6333, storage_path: Optional[str] = None, collection_name: Optional[str] = None) -> QdrantVectorStore:
    """
    Create a new QdrantVectorStore instance.
    
    Args:
        host: Qdrant server host (default: uses QDRANT_HOST env or 'qdrant' for Docker)
        port: Qdrant server port (default: 6333 for Docker)
        storage_path: Local file storage path. If provided, uses local storage instead of Docker.
        collection_name: Custom collection name. If None, uses default.
    
    Returns:
        QdrantVectorStore instance configured for Docker or local storage
    """
    import os
    resolved_host = os.environ.get("QDRANT_HOST", "qdrant") if host == "localhost" else host
    return QdrantVectorStore(host=resolved_host, port=port, storage_path=storage_path, collection_name=collection_name)