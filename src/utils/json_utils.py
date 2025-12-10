"""
JSON Utilities Module

Utilities for JSON serialization with NumPy type handling.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_numpy_to_json(obj: Any) -> Any:
    """
    Recursively convert numpy arrays and types to JSON-serializable format.
    
    Args:
        obj: Object to convert (can be dict, list, or numpy type)
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json(item) for item in obj]
    else:
        return obj


def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file with NumPy type handling.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, cls=NumpyEncoder)


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Load JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
