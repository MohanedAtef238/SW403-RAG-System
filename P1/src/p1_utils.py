"""
P1-specific imports and utilities.
Creates type union for P1 and P2 chunk compatibility.
"""

from typing import Union, List, Protocol, Sequence, Any
from .chunking import FunctionChunk
import numpy as np


# Protocol for chunk compatibility (now only FunctionChunk for P1)
class ChunkProtocol(Protocol):
    file_path: str
    function_name: str
    start_line: int
    def to_payload(self) -> dict:
        ...

# Type alias for chunk (only FunctionChunk)
ChunkType = FunctionChunk

def create_chunk_compatible_vector_store(collection_name: str = "functions_p1"):
    """Create vector store that accepts both P1 and P2 chunks."""
    from .vector_store import QdrantVectorStore
    
    class P1CompatibleVectorStore(QdrantVectorStore):
        """Vector store that accepts both P1FunctionChunk and FunctionChunk."""
        
        def upsert_chunks(self, chunks: Sequence[Any], embeddings: np.ndarray) -> bool:
            """Upsert chunks of either P1 or P2 type."""
            if len(chunks) != len(embeddings):
                raise ValueError("Number of chunks must match number of embeddings")
            
            try:
                from qdrant_client.models import PointStruct
                import time
                
                # Create points for upsert
                points = []
                for i, chunk in enumerate(chunks):
                    # Generate deterministic positive ID
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
                
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Upserted {len(points)} function chunks in {upsert_time:.3f}s")
                return True
                
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error upserting chunks: {e}")
                return False
    
    return P1CompatibleVectorStore(collection_name=collection_name)