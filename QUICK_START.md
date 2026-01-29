# Quick Start Guide - MEDAL Framework

This guide helps you get started with the refactored MEDAL framework.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/johndmendonca/medal.git
   cd medal
   ```

2. **Install dependencies:**
   ```bash
   # Using conda (recommended)
   conda env create -f environment.yml
   conda activate medal
   
   # Or using pip
   pip install -r requirements.txt
   
   # Install package in development mode
   pip install -e .
   ```

3. **Set up environment variables:**
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export GEMENI_KEY="your_google_gemini_api_key"
   export OPENROUTER_KEY="your_openrouter_api_key"
   ```

4. **Create configuration file:**
   ```bash
   cp config.example.yaml config.yaml
   # Edit config.yaml with your settings
   ```

## Basic Usage

### Using Python API

```python
from medal.tasks.dialogue_generation import DialogueGenerator
from medal.config import Config

# Load configuration
config = Config.from_yaml('config.yaml')

# Generate dialogue turn
generator = DialogueGenerator(
    context="path/to/dialogue/dataset",
    lang="english",
    model="meta-llama/Llama-3.3-70B-Instruct",
    role="user",
    turn=1,
    run_id="vanilla",
    config=config
)

# Generate requests
generator.generate()

# Evaluate responses
generator.evaluate()

# Regenerate failed responses
regens_needed = generator.regenerate()
print(f"Regenerating {regens_needed} examples")
```

### Using CLI

```bash
# Generate dialogue turn
medal dialogue path/to/context \
    --lang english \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --role user \
    --turn 1 \
    --type generate

# Evaluate dialogue turn
medal dialogue path/to/context \
    --lang english \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --role user \
    --turn 1 \
    --type evaluate

# Process evaluations and regenerate
medal dialogue path/to/context \
    --lang english \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --role user \
    --turn 1 \
    --type process
```

### Using Original Shell Scripts

The original shell scripts are still supported:

```bash
# Generate starters
./0_generate_starters_vllm.sh google gemma-3-27b-it english 4 ATOMIC10X_persona_1k_3

# Generate dialogue turn
./4_generate_turnX.sh google gemma-3-27b-it english 1 meta-llama/Llama-3.3-70B-Instruct 4 user
```

## Configuration

Edit `config.yaml` to customize:

```yaml
paths:
  batches_to_process: "batches_to_process"
  completed_batches: "completed_batches"
  dialogues: "dialogues"

model:
  temperature: 0.9
  top_p: 0.95
  max_tokens: 512

api:
  provider: "openai"
  max_requests_per_minute: 1000.0
  max_tokens_per_minute: 100000.0
```

## Workflow Example

1. **Generate narrative starters:**
   ```python
   from medal.tasks.narrative_generation import NarrativeGenerator
   
   generator = NarrativeGenerator(
       dataset="tasks/narrative_generation/data/ATOMIC10X_persona_1k_3.json",
       lang="english",
       model="meta-llama/Llama-3.3-70B-Instruct",
       run_id="vanilla"
   )
   generator.generate()
   ```

2. **Process batch with VLLM:**
   ```bash
   python agents/vllm_batch.py \
       --input_file batches_to_process/vanilla/ATOMIC10X_persona_1k_3_english_turn0_Llama-3.3-70B-Instruct.jsonl \
       --tensor_parallel_size 4
   ```

3. **Evaluate responses:**
   ```python
   generator.evaluate()
   ```

4. **Process evaluations:**
   ```python
   regens = generator.regenerate()
   while regens > 0:
       # Process regenerations...
       regens = generator.regenerate()
   ```

## Key Utilities

### Path Management

```python
from medal.utils import build_batch_path, build_dialogue_path

# Build batch file path
batch_path = build_batch_path(
    lang="english",
    run_id="vanilla",
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    dataset_name="ATOMIC10X_persona_1k_0",
    turn=0,
    file_type="gen"
)

# Build dialogue dataset path
dialogue_path = build_dialogue_path(
    lang="english",
    run_id="vanilla",
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    turn=0
)
```

### I/O Operations

```python
from medal.utils import load_jsonl, save_jsonl, load_dataset

# Load JSONL file
data = load_jsonl("path/to/file.jsonl")

# Save JSONL file
save_jsonl(data, "path/to/output.jsonl")

# Load HuggingFace dataset
dataset = load_dataset("path/to/dataset")
```

## Next Steps

- Read [REFACTORING.md](REFACTORING.md) for detailed refactoring information
- Check the original [README.md](readme.md) for paper and dataset information
- Explore the `medal/` directory for more examples
