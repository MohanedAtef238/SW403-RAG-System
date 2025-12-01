"""
P2 AST-based code chunking module for semantic function splitting.
Extracts functions with signature preservation and rich metadata using AST parsing.
This is the "Smarter Chunks" implementation.
"""

import ast
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FunctionChunk:
    """Container for a function chunk with comprehensive metadata."""
    
    def __init__(
        self,
        file_path: str,
        function_name: str,
        function_signature: str,
        start_line: int,
        end_line: int,
        original_chunk_text: str,
        docstring: Optional[str] = None,
        decorators: Optional[List[str]] = None,
        ast_node_type: str = "function",
        complexity_score: int = 1
    ):
        self.file_path = str(Path(file_path).resolve())
        self.relative_path = self._get_relative_path(file_path)
        self.function_name = function_name
        self.function_signature = function_signature
        self.start_line = start_line
        self.end_line = end_line
        self.original_chunk_text = original_chunk_text
        self.docstring = docstring
        self.decorators = decorators or []
        self.ast_node_type = ast_node_type
        self.complexity_score = complexity_score
        self.file_extension = Path(file_path).suffix
    
    def _get_relative_path(self, file_path: str) -> str:
        """Get project-relative path."""
        try:
            # Try to get relative to current working directory
            return str(Path(file_path).relative_to(Path.cwd()))
        except ValueError:
            # If not relative to cwd, just use the filename
            return Path(file_path).name
    
    def to_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload format."""
        return {
            "file_path": self.file_path,
            "relative_path": self.relative_path,
            "function_signature": self.function_signature,
            "line_numbers": {
                "start": self.start_line,
                "end": self.end_line
            },
            "function_name": self.function_name,
            "original_chunk_text": self.original_chunk_text,
            "metadata": {
                "file_extension": self.file_extension,
                "ast_node_type": self.ast_node_type,
                "complexity_score": self.complexity_score,
                "docstring": self.docstring,
                "decorators": self.decorators
            }
        }
    
    def __repr__(self) -> str:
        return f"FunctionChunk({self.function_name} @ {self.relative_path}:{self.start_line}-{self.end_line})"


class CodeChunker:
    """AST-based code chunker for extracting functions."""
    
    def __init__(self):
        self.chunks: List[FunctionChunk] = []
    
    def extract_function_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Extract clean function signature from AST node."""
        args = []
        
        # Regular arguments with type annotations and defaults
        for i, arg in enumerate(node.args.args):
            arg_str = arg.arg
            
            # Add type annotation if present
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            
            # Add default value if present
            defaults_offset = len(node.args.args) - len(node.args.defaults)
            if i >= defaults_offset:
                default_idx = i - defaults_offset
                arg_str += f" = {ast.unparse(node.args.defaults[default_idx])}"
            
            args.append(arg_str)
        
        # Handle *args
        if node.args.vararg:
            vararg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                vararg_str += f": {ast.unparse(node.args.vararg.annotation)}"
            args.append(vararg_str)
        
        # Handle keyword-only arguments
        for i, arg in enumerate(node.args.kwonlyargs):
            kwarg_str = arg.arg
            if arg.annotation:
                kwarg_str += f": {ast.unparse(arg.annotation)}"
            if i < len(node.args.kw_defaults):
                default_val = node.args.kw_defaults[i]
                if default_val is not None:
                    kwarg_str += f" = {ast.unparse(default_val)}"
            args.append(kwarg_str)
        
        # Handle **kwargs
        if node.args.kwarg:
            kwarg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                kwarg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
            args.append(kwarg_str)
        
        # Handle return annotation
        return_annotation = ""
        if node.returns:
            return_annotation = f" -> {ast.unparse(node.returns)}"
        
        # Build signature
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(args)}){return_annotation}:"
    
    def extract_docstring(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[str]:
        """Extract docstring from function node."""
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            return node.body[0].value.value
        return None
    
    def extract_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
        """Extract decorator names from function node."""
        decorators = []
        for decorator in node.decorator_list:
            try:
                decorators.append(ast.unparse(decorator))
            except Exception:
                # Fallback for complex decorators
                decorators.append(str(decorator))
        return decorators
    
    def calculate_complexity(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        """Calculate rough cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Decision points increase complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def chunk_file(self, file_path: str, source_code: Optional[str] = None) -> List[FunctionChunk]:
        """
        Extract function chunks from a Python file.
        
        Args:
            file_path: Path to the Python file
            source_code: Optional source code string (if not provided, reads from file)
            
        Returns:
            List of FunctionChunk objects
        """
        try:
            if source_code is None:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
            
            # Parse the AST
            tree = ast.parse(source_code)
            source_lines = source_code.split('\n')
            
            chunks = []
            
            # Walk through all nodes to find function definitions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Extract function text
                    start_line = node.lineno
                    end_line = node.end_lineno if node.end_lineno else start_line
                    
                    # Get the actual function text
                    function_lines = source_lines[start_line-1:end_line]
                    original_text = '\n'.join(function_lines)
                    
                    # Create chunk
                    chunk = FunctionChunk(
                        file_path=file_path,
                        function_name=node.name,
                        function_signature=self.extract_function_signature(node),
                        start_line=start_line,
                        end_line=end_line,
                        original_chunk_text=original_text,
                        docstring=self.extract_docstring(node),
                        decorators=self.extract_decorators(node),
                        ast_node_type="async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                        complexity_score=self.calculate_complexity(node)
                    )
                    
                    chunks.append(chunk)
            
            logger.info(f"Extracted {len(chunks)} functions from {Path(file_path).name}")
            return chunks
            
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error chunking {file_path}: {e}")
            return []
    
    def chunk_directory(self, directory_path: str, pattern: str = "*.py") -> List[FunctionChunk]:
        """
        Extract function chunks from all Python files in a directory.
        
        Args:
            directory_path: Path to directory
            pattern: File pattern to match (default: "*.py")
            
        Returns:
            List of all FunctionChunk objects from the directory
        """
        directory = Path(directory_path)
        all_chunks = []
        
        if not directory.exists():
            logger.error(f"Directory {directory_path} does not exist")
            return []
        
        # Find all Python files
        python_files = list(directory.rglob(pattern))
        
        for file_path in python_files:
            # Skip __pycache__ and virtual environment files
            if '__pycache__' in str(file_path) or '.venv' in str(file_path):
                continue
            
            chunks = self.chunk_file(str(file_path))
            all_chunks.extend(chunks)
        
        logger.info(f"Extracted {len(all_chunks)} total functions from {len(python_files)} files in {directory_path}")
        return all_chunks
    
    def chunk_multiple_files(self, file_paths: List[str]) -> List[FunctionChunk]:
        """
        Extract function chunks from multiple files.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of all FunctionChunk objects
        """
        all_chunks = []
        
        for file_path in file_paths:
            chunks = self.chunk_file(file_path)
            all_chunks.extend(chunks)
        
        logger.info(f"Extracted {len(all_chunks)} total functions from {len(file_paths)} files")
        return all_chunks


def create_chunker() -> CodeChunker:
    """Create a new CodeChunker instance."""
    return CodeChunker()


# Example usage and testing
if __name__ == "__main__":
    # Test with the current main.py
    chunker = create_chunker()
    chunks = chunker.chunk_file("../main.py")
    
    for chunk in chunks:
        print(f"Function: {chunk.function_name}")
        print(f"Signature: {chunk.function_signature}")
        print(f"Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"Complexity: {chunk.complexity_score}")
        print("-" * 50)