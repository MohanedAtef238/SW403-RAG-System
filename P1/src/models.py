"""
Model management module with caching and performance monitoring.
Handles global initialization of all-MiniLM-L6-v2 for fast inference.
"""

import os
import time
import psutil
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model configuration
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32  # Configurable batch size for fine-tuning
MAX_TOKENS = 400  # Token limit for function text

# Global model instance (cached on startup)
_model: Optional[SentenceTransformer] = None
_model_memory_baseline: float = 0.0
_process = psutil.Process()


class ModelMetrics:
    """Container for model performance metrics."""
    
    def __init__(self):
        self.embedding_time: float = 0.0
        self.memory_usage: float = 0.0
        self.batch_size: int = 0
        self.text_count: int = 0


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    return _process.memory_info().rss / 1024 / 1024


def initialize_model() -> None:
    """Initialize the global embedding model with memory tracking."""
    global _model, _model_memory_baseline
    
    if _model is not None:
        logger.info("Model already initialized")
        return
    
    logger.info(f"Initializing {MODEL_NAME} model...")
    memory_before = get_memory_usage_mb()
    
    start_time = time.time()
    _model = SentenceTransformer(MODEL_NAME)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        _model = _model.to('cuda')
        logger.info("Model moved to GPU")
    else:
        logger.info("Model running on CPU")
    
    # Warm up the model with a dummy inference
    _model.encode(["def dummy_function(): pass"], convert_to_numpy=True)
    
    initialization_time = time.time() - start_time
    memory_after = get_memory_usage_mb()
    _model_memory_baseline = memory_after - memory_before
    
    logger.info(f"Model initialized in {initialization_time:.2f}s")
    logger.info(f"Model memory footprint: {_model_memory_baseline:.1f} MB")


def get_model() -> SentenceTransformer:
    """Get the cached model instance."""
    if _model is None:
        raise RuntimeError("Model not initialized. Call initialize_model() first.")
    return _model


def prepare_function_text(function_text: str, max_tokens: int = MAX_TOKENS) -> str:
    """
    Prepare function text for embedding with intelligent truncation.
    Preserves signature + docstring, truncates body if needed.
    """
    lines = function_text.strip().split('\n')
    if not lines:
        return ""
    
    # Always preserve the function signature
    signature_line = lines[0]
    if len(lines) == 1:
        return signature_line
    
    # Look for docstring
    docstring_lines = []
    body_start_idx = 1
    
    # Check if second line starts a docstring
    if len(lines) > 1:
        second_line = lines[1].strip()
        if second_line.startswith('"""') or second_line.startswith("'''"):
            quote_type = '"""' if second_line.startswith('"""') else "'''"
            
            # Single-line docstring
            if second_line.count(quote_type) >= 2:
                docstring_lines.append(lines[1])
                body_start_idx = 2
            else:
                # Multi-line docstring
                docstring_lines.append(lines[1])
                for i in range(2, len(lines)):
                    docstring_lines.append(lines[i])
                    if quote_type in lines[i]:
                        body_start_idx = i + 1
                        break
    
    # Combine essential parts
    essential_parts = [signature_line] + docstring_lines
    essential_text = '\n'.join(essential_parts)
    
    # Rough token estimation (1 token ≈ 4 characters for code)
    essential_tokens = len(essential_text) // 4
    remaining_token_budget = max_tokens - essential_tokens
    
    # Add body lines if budget allows
    if remaining_token_budget > 0 and body_start_idx < len(lines):
        body_lines = lines[body_start_idx:]
        
        # Add body lines until token budget is exhausted
        included_body_lines = []
        current_tokens = 0
        
        for line in body_lines:
            line_tokens = len(line) // 4
            if current_tokens + line_tokens <= remaining_token_budget:
                included_body_lines.append(line)
                current_tokens += line_tokens
            else:
                break
        
        if included_body_lines:
            essential_parts.extend(included_body_lines)
    
    return '\n'.join(essential_parts)


def encode_texts(texts: List[str], batch_size: int = BATCH_SIZE) -> tuple[np.ndarray, ModelMetrics]:
    """
    Encode texts into embeddings with performance tracking.
    
    Args:
        texts: List of text strings to encode
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (embeddings array, metrics)
    """
    if not texts:
        return np.array([]), ModelMetrics()
    
    model = get_model()
    metrics = ModelMetrics()
    
    # Prepare texts
    prepared_texts = [prepare_function_text(text) for text in texts]
    
    # Track memory before encoding
    memory_before = get_memory_usage_mb()
    
    # Encode with timing
    start_time = time.time()
    embeddings = model.encode(
        prepared_texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 10,  # Show progress for larger batches
        normalize_embeddings=True,  # Important for cosine similarity
        convert_to_numpy=True
    )
    encoding_time = time.time() - start_time
    
    # Track memory after encoding
    memory_after = get_memory_usage_mb()
    
    # Populate metrics
    metrics.embedding_time = encoding_time
    metrics.memory_usage = memory_after - memory_before + _model_memory_baseline
    metrics.batch_size = batch_size
    metrics.text_count = len(texts)
    
    logger.info(f"Encoded {len(texts)} texts in {encoding_time:.3f}s "
               f"(batch_size={batch_size}, {len(texts)/encoding_time:.1f} texts/sec)")
    
    return embeddings, metrics


def encode_single_text(text: str) -> tuple[np.ndarray, ModelMetrics]:
    """Encode a single text with performance tracking."""
    return encode_texts([text], batch_size=1)


def get_model_info() -> Dict[str, Any]:
    """Get model information and current memory usage."""
    return {
        "model_name": MODEL_NAME,
        "model_dimensions": 384,
        "batch_size": BATCH_SIZE,
        "max_tokens": MAX_TOKENS,
        "memory_baseline_mb": _model_memory_baseline,
        "current_memory_mb": get_memory_usage_mb(),
        "device": "cuda" if torch.cuda.is_available() and _model is not None else "cpu",
        "initialized": _model is not None
    }