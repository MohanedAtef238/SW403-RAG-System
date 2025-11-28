# Sample Python file for testing both P1 and P2 implementations
from typing import Optional, Dict, List, Any
import asyncio

def simple_function():
    """A simple function with no parameters."""
    return "hello world"

def function_with_params(x: int, y: str = "default") -> bool:
    """Function with type hints and default parameters."""
    if x > 0:
        return True
    return False

@property 
def decorated_function():
    """Function with a decorator."""
    return "decorated"

async def async_function(*args, **kwargs) -> Dict[str, Any]:
    """Async function with variable arguments."""
    await asyncio.sleep(0.1)
    return {"args": args, "kwargs": kwargs}

class SampleClass:
    """A sample class with methods."""
    
    def __init__(self, value: int):
        self.value = value
    
    def simple_method(self):
        """Simple method without parameters."""
        return self.value
        
    def complex_method(self, data: Optional[Dict[str, List[Any]]] = None):
        """Complex method with nested type annotations."""
        if data:
            for key, values in data.items():
                if isinstance(values, list) and len(values) > 0:
                    return key
        return None
    
    @staticmethod
    def static_method(param: str) -> str:
        """Static method with type annotations."""
        return param.upper()
    
    @classmethod
    def class_method(cls, instance_count: int):
        """Class method example."""
        return [cls(i) for i in range(instance_count)]

# Function with complex logic for complexity testing
def complex_logic_function(items: List[Dict[str, Any]]) -> Optional[str]:
    """Function with complex control flow for testing cyclomatic complexity."""
    if not items:
        return None
    
    for item in items:
        if "type" in item:
            if item["type"] == "A":
                try:
                    if "value" in item and item["value"] > 10:
                        return "high_value_A"
                    elif "value" in item:
                        return "low_value_A"
                except (KeyError, TypeError):
                    continue
            elif item["type"] == "B":
                if item.get("active", False):
                    return "active_B"
                else:
                    return "inactive_B"
        else:
            continue
    
    return "no_match"