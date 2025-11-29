
"""
P1 Baseline code chunking module for function-level splitting.
Extracts functions with signature and docstring only.
"""


import ast
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)



class FunctionChunk:
    """Container for a function chunk (baseline: only essential info)."""
    def __init__(self, file_path: str, function_name: str, function_signature: str, start_line: int, end_line: int, original_chunk_text: str, docstring: Optional[str] = None):
        self.file_path = str(Path(file_path).resolve())
        self.function_name = function_name
        self.function_signature = function_signature
        self.start_line = start_line
        self.end_line = end_line
        self.original_chunk_text = original_chunk_text
        self.docstring = docstring

    def to_payload(self):
        return {
            "file_path": self.file_path,
            "function_name": self.function_name,
            "function_signature": self.function_signature,
            "line_numbers": {
                "start": self.start_line,
                "end": self.end_line
            },
            "original_chunk_text": self.original_chunk_text,
            "docstring": self.docstring
        }

    def __repr__(self):
        return f"FunctionChunk({self.function_name} @ {self.file_path}:{self.start_line}-{self.end_line})"



class CodeChunker:
    """Baseline code chunker for extracting functions only."""
    def chunk_file(self, file_path: str, source_code: Optional[str] = None) -> List[FunctionChunk]:
        try:
            if source_code is None:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
            tree = ast.parse(source_code)
            source_lines = source_code.split('\n')
            chunks = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start_line
                    function_lines = source_lines[start_line-1:end_line]
                    original_text = '\n'.join(function_lines)
                    signature = self.extract_function_signature(node)
                    docstring = self.extract_docstring(node)
                    chunk = FunctionChunk(
                        file_path=file_path,
                        function_name=node.name,
                        function_signature=signature,
                        start_line=start_line,
                        end_line=end_line,
                        original_chunk_text=original_text,
                        docstring=docstring
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

    def extract_function_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        args = [arg.arg for arg in node.args.args]
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(args)}):"

    def extract_docstring(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[str]:
        if (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
            return node.body[0].value.value
        return None

def create_chunker() -> CodeChunker:
    return CodeChunker()