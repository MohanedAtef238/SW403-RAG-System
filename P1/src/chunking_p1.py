"""
P1 Baseline Chunking: Simple text-based function splitting.
This is the true "baseline" approach using basic text patterns.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class P1FunctionChunk:
    """Simple function chunk for P1 baseline approach."""
    
    def __init__(
        self,
        file_path: str,
        function_name: str,
        start_line: int,
        end_line: int,
        raw_text: str
    ):
        self.file_path = str(Path(file_path).resolve())
        self.relative_path = self._get_relative_path(file_path)
        self.function_name = function_name
        self.start_line = start_line
        self.end_line = end_line
        self.raw_text = raw_text
        self.file_extension = Path(file_path).suffix
    
    def _get_relative_path(self, file_path: str) -> str:
        """Get project-relative path."""
        try:
            return str(Path(file_path).relative_to(Path.cwd()))
        except ValueError:
            return Path(file_path).name
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert to vector store payload format."""
        return {
            "file_path": self.file_path,
            "relative_path": self.relative_path,
            "function_name": self.function_name,
            "line_numbers": {
                "start": self.start_line,
                "end": self.end_line
            },
            "raw_text": self.raw_text,
            "metadata": {
                "file_extension": self.file_extension,
                "chunking_method": "P1_text_based",
                "has_ast_info": False
            }
        }


class P1TextChunker:
    """P1 Baseline: Simple regex-based function extraction."""
    
    def __init__(self):
        # Simple regex pattern for function definitions
        self.function_pattern = re.compile(
            r'^(\s*)(def|async def)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            re.MULTILINE
        )
    
    def chunk_file(self, file_path: str, source_code: Optional[str] = None) -> List[P1FunctionChunk]:
        """
        Extract functions using simple text patterns (P1 baseline).
        
        Args:
            file_path: Path to Python file
            source_code: Optional source code (reads from file if None)
            
        Returns:
            List of P1FunctionChunk objects
        """
        if source_code is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                return []
        
        lines = source_code.split('\n')
        chunks = []
        
        # Find all function definitions
        matches = list(self.function_pattern.finditer(source_code))
        
        for i, match in enumerate(matches):
            start_pos = match.start()
            function_name = match.group(3)
            
            # Find line number of function start
            start_line = source_code[:start_pos].count('\n') + 1
            
            # Determine function end using simple indentation logic
            if i + 1 < len(matches):
                # Next function exists - end before it starts
                next_start_pos = matches[i + 1].start()
                end_line = source_code[:next_start_pos].count('\n')
            else:
                # Last function - go to end of file
                end_line = len(lines)
            
            # Extract function text
            function_lines = lines[start_line-1:end_line]
            
            # Simple indentation-based trimming
            if function_lines:
                base_indent = len(function_lines[0]) - len(function_lines[0].lstrip())
                
                # Find actual end by looking for next unindented line
                actual_end = len(function_lines)
                for j in range(1, len(function_lines)):
                    line = function_lines[j]
                    if line.strip() and len(line) - len(line.lstrip()) <= base_indent:
                        actual_end = j
                        break
                
                function_text = '\n'.join(function_lines[:actual_end])
                actual_end_line = start_line + actual_end - 1
                
                chunk = P1FunctionChunk(
                    file_path=file_path,
                    function_name=function_name,
                    start_line=start_line,
                    end_line=actual_end_line,
                    raw_text=function_text
                )
                chunks.append(chunk)
        
        logger.info(f"P1: Extracted {len(chunks)} functions from {Path(file_path).name}")
        return chunks


def create_p1_chunker() -> P1TextChunker:
    """Factory function for P1 chunker."""
    return P1TextChunker()


# Test function
if __name__ == "__main__":
    chunker = create_p1_chunker()
    chunks = chunker.chunk_file("../main.py")
    
    print(f"Found {len(chunks)} functions:")
    for chunk in chunks:
        print(f"- {chunk.function_name} ({chunk.start_line}-{chunk.end_line})")
        print(f"  Text preview: {chunk.raw_text[:100]}...")