"""
Utility functions for path management.
"""
from pathlib import Path
from typing import Tuple, Optional


def extract_path_components(file_path: str, depth: int = 3) -> Tuple[str, ...]:
    """
    Extract directory components from a file path.
    
    Args:
        file_path: Path to extract components from
        depth: Number of directory levels to extract (default: 3)
    
    Returns:
        Tuple of directory names from deepest to shallowest
    """
    path = Path(file_path)
    components = []
    
    for _ in range(depth):
        if path.parent == path:  # Reached root
            break
        components.append(path.parent.name)
        path = path.parent
    
    # Pad with empty strings if needed
    while len(components) < depth:
        components.append("")
    
    return tuple(reversed(components))


def build_output_path(
    base_dir: str,
    *components: str,
    filename: Optional[str] = None,
    extension: Optional[str] = None
) -> Path:
    """
    Build an output path from components.
    
    Args:
        base_dir: Base directory for the path
        *components: Additional path components
        filename: Optional filename (without extension)
        extension: Optional file extension (without dot)
    
    Returns:
        Path object for the constructed path
    """
    path = Path(base_dir)
    for component in components:
        if component:
            path = path / component
    
    if filename:
        if extension:
            filename = f"{filename}.{extension}"
        path = path / filename
    
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_batch_paths(
    input_file: str,
    base_dir: str,
    output_type: str = "completed"
) -> dict:
    """
    Generate standard batch file paths based on input file location.
    
    Args:
        input_file: Path to input file
        base_dir: Base directory (batches_to_process, completed_batches, etc.)
        output_type: Type of output ('completed', 'submitted', 'to_process')
    
    Returns:
        Dictionary with path components and full paths
    """
    path = Path(input_file)
    input_name = path.stem
    
    # Extract directory structure
    three_up, two_up, one_up = extract_path_components(input_file, depth=3)
    
    # Build output directory
    if output_type == "completed":
        output_dir = Path(base_dir) / three_up / two_up / one_up
    elif output_type == "submitted":
        output_dir = Path(base_dir) / three_up / two_up / one_up
    else:
        output_dir = Path(base_dir) / three_up / two_up / one_up
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "input_name": input_name,
        "output_dir": str(output_dir),
        "output_file": str(output_dir / f"{input_name}.jsonl"),
        "components": {
            "three_up": three_up,
            "two_up": two_up,
            "one_up": one_up,
        }
    }


def build_dialogue_path(
    lang: str,
    run_id: str,
    model_name: str,
    turn: int,
    base_dir: str = "dialogues"
) -> Path:
    """
    Build a dialogue dataset path.
    
    Args:
        lang: Language code
        run_id: Run identifier
        model_name: Model name (will extract base name if contains '/')
        turn: Turn number
        base_dir: Base directory for dialogues
    
    Returns:
        Path to dialogue dataset
    """
    model_base = model_name.split("/")[-1] if "/" in model_name else model_name
    path = Path(base_dir) / lang / run_id / model_base / f"turn-{turn}"
    return path


def build_batch_path(
    lang: str,
    run_id: str,
    model_name: str,
    dataset_name: str,
    turn: int,
    file_type: str = "gen",
    base_dir: str = "batches_to_process"
) -> Path:
    """
    Build a batch file path.
    
    Args:
        lang: Language code
        run_id: Run identifier
        model_name: Model name
        dataset_name: Dataset name
        turn: Turn number
        file_type: Type of file ('gen', 'eval', 'regen')
        base_dir: Base directory
    
    Returns:
        Path to batch file
    """
    model_base = model_name.split("/")[-1] if "/" in model_name else model_name
    
    if turn == 0:
        filename = f"{dataset_name}_{lang}_turn{turn}_{model_base}"
    else:
        filename = f"turn-{turn}_{model_base}"
    
    if file_type == "eval":
        filename += "_eval"
    elif file_type == "regen":
        filename += "_regen"
    
    path = Path(base_dir) / run_id / f"{filename}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_dialogue_batch_path(
    lang: str,
    run_id: str,
    model_name: str,
    turn: int,
    file_type: str = "gen",
    base_dir: str = "batches_to_process",
) -> Path:
    """
    Build a batch file path for dialogue turns (lang/run_id/model structure).

    Args:
        lang: Language code.
        run_id: Run identifier (e.g. model_user or user_model).
        model_name: Model name (e.g. meta-llama/Llama-3.3-70B-Instruct).
        turn: Turn number.
        file_type: 'gen', 'eval', or 'regen'.
        base_dir: Base directory for batches.

    Returns:
        Path to the batch file.
    """
    model_base = model_name.split("/")[-1] if "/" in model_name else model_name
    filename = f"turn-{turn}_{model_base}"
    if file_type == "eval":
        filename += "_eval"
    elif file_type == "regen":
        filename += "_regen"
    path = Path(base_dir) / lang / run_id / model_base / f"{filename}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
