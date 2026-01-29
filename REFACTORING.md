# MEDAL Framework Refactoring Guide

This document describes the refactoring improvements made to the MEDAL codebase to improve readability, maintainability, and job-readiness.

## Overview

The refactoring focuses on:
1. **Better code organization** - Clear module structure and separation of concerns
2. **Configuration management** - YAML-based configuration instead of hardcoded values
3. **Utility functions** - Reusable path and I/O utilities
4. **Type hints** - Better type safety and IDE support
5. **Error handling** - Improved error messages and handling
6. **CLI interface** - Unified command-line interface
7. **Documentation** - Better docstrings and code comments

## New Structure

```
medal/
├── __init__.py          # Package initialization
├── config.py            # Configuration management
├── prompts.py           # All prompt templates
├── cli.py               # Command-line interface
├── core/
│   ├── __init__.py
│   └── api_client.py    # API client abstractions
├── utils/
│   ├── __init__.py
│   ├── paths.py         # Path utility functions
│   └── io.py            # I/O utility functions
└── tasks/
    ├── __init__.py
    ├── dialogue_generation.py    # Refactored dialogue generation
    ├── narrative_generation.py   # (To be refactored)
    └── dialogue_evaluation.py    # (To be refactored)
```

## Key Improvements

### 1. Configuration Management

**Before:**
- Hardcoded values scattered throughout code
- Environment variables accessed directly
- No centralized configuration

**After:**
- YAML-based configuration file (`config.yaml`)
- `Config` dataclass with nested configs
- Environment variable fallbacks

**Usage:**
```python
from medal.config import Config

# Load from YAML
config = Config.from_yaml('config.yaml')

# Or use defaults
config = Config()
```

### 2. Path Utilities

**Before:**
- Repeated path construction logic
- Hardcoded directory structures
- Inconsistent path handling

**After:**
- Centralized path utilities in `medal.utils.paths`
- Consistent path building functions
- Automatic directory creation

**Usage:**
```python
from medal.utils import build_batch_path, build_dialogue_path

# Build batch path
batch_path = build_batch_path(
    lang="english",
    run_id="vanilla",
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    dataset_name="ATOMIC10X_persona_1k_0",
    turn=0,
    file_type="gen"
)
```

### 3. I/O Utilities

**Before:**
- Repeated JSONL/JSON loading code
- Inconsistent error handling
- Manual file operations

**After:**
- Centralized I/O functions in `medal.utils.io`
- Consistent error handling
- Automatic directory creation

**Usage:**
```python
from medal.utils import load_jsonl, save_jsonl, load_dataset

# Load JSONL
data = load_jsonl("path/to/file.jsonl")

# Save JSONL
save_jsonl(data, "path/to/output.jsonl")

# Load dataset
dataset = load_dataset("path/to/dataset")
```

### 4. Refactored Dialogue Generator

**Before:**
- Tightly coupled to argparse
- Hardcoded paths
- Mixed concerns

**After:**
- Clean class-based API
- Dependency injection
- Better separation of concerns

**Usage:**
```python
from medal.tasks.dialogue_generation import DialogueGenerator
from medal.config import Config

config = Config()
generator = DialogueGenerator(
    context="path/to/dialogue/dataset",
    lang="english",
    model="meta-llama/Llama-3.3-70B-Instruct",
    role="user",
    turn=1,
    run_id="vanilla",
    config=config
)

generator.generate()
```

### 5. Command-Line Interface

**Before:**
- Multiple shell scripts
- Complex argument parsing
- Hard to use programmatically

**After:**
- Unified CLI using Click
- Better help messages
- Can be used as Python API

**Usage:**
```bash
# Generate dialogue turn
medal dialogue path/to/context --lang english --model meta-llama/Llama-3.3-70B-Instruct \
    --role user --turn 1 --type generate

# Evaluate dialogues
medal evaluate path/to/dialogues --lang english --model gemini-2.0-flash
```

## Migration Guide

### For Existing Scripts

1. **Update imports:**
   ```python
   # Old
   import orjson
   from datasets import load_from_disk
   
   # New
   from medal.utils import load_jsonl, load_dataset
   ```

2. **Use configuration:**
   ```python
   # Old
   temperature = 0.9
   max_tokens = 512
   
   # New
   from medal.config import Config
   config = Config.from_yaml('config.yaml')
   temperature = config.model.temperature
   max_tokens = config.model.max_tokens
   ```

3. **Use path utilities:**
   ```python
   # Old
   out_dir = f'{lang}/{run_id}/{model.split("/")[-1]}/'
   os.makedirs(f"batches_to_process/{out_dir}", exist_ok=True)
   
   # New
   from medal.utils import build_batch_path
   batch_path = build_batch_path(lang, run_id, model, dataset, turn)
   ```

## Benefits

1. **Readability**: Clear structure and consistent naming
2. **Maintainability**: Centralized utilities reduce duplication
3. **Testability**: Better separation of concerns enables unit testing
4. **Extensibility**: Easy to add new features or providers
5. **Job-readiness**: Professional structure and documentation

## Next Steps

1. Refactor remaining modules (`narrative_generation`, `dialogue_evaluation`)
2. Add unit tests
3. Add type checking with mypy
4. Add logging throughout
5. Create comprehensive documentation
6. Add example notebooks

## Notes on LangChain

**Decision: Not adding LangChain**

After reviewing the codebase, LangChain would add unnecessary complexity:
- The codebase uses direct API calls which is more efficient
- LangChain's abstractions don't match the batch processing needs
- The current approach is more transparent and easier to debug
- For job interviews, showing direct API usage demonstrates better understanding

Instead, we've created clean abstractions (`BaseAPIClient`, `Config`) that provide structure without heavy dependencies.
