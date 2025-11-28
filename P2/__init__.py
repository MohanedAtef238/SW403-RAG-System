"""
SW403 P1 RAG System - Main entry point and CLI utilities.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Now we can import our modules
from src.api import app


def main():
    """Main entry point for development."""
    import uvicorn
    print("Starting SW403 P1 RAG System...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()