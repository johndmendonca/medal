"""
I/O utility functions for reading and writing data files.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import orjson
from datasets import Dataset, load_from_disk


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List of dictionaries parsed from JSONL lines
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [orjson.loads(line) for line in f if line.strip()]


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to output JSONL file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(orjson.dumps(item).decode('utf-8') + '\n')


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Dictionary parsed from JSON
    """
    import json5
    with open(file_path, 'r', encoding='utf-8') as f:
        return json5.load(f)


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to output JSON file
        indent: JSON indentation level
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_dataset(path: str) -> Dataset:
    """
    Load a HuggingFace dataset from disk.
    
    Args:
        path: Path to dataset directory
    
    Returns:
        Dataset object
    """
    return load_from_disk(path)


def save_dataset(dataset: Dataset, path: str) -> None:
    """
    Save a HuggingFace dataset to disk.
    
    Args:
        dataset: Dataset to save
        path: Path to save dataset
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(path)
