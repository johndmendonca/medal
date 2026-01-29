# Example Scripts

These scripts mirror the shell workflow for full dialogue generation using the refactored `medal` package.

## Prerequisites

- Install the project: `pip install -e .` (from repo root)
- Set environment variables: `OPENAI_API_KEY`, `GEMENI_KEY` (for evaluation), and optionally `OPENROUTER_KEY`
- For VLLM: GPU with enough memory for the chosen model

## generate_dialogue_vllm.py

Generates full multilingual dialogues using VLLM for both user (starter + turns) and assistant. Equivalent to `./generate_dialogues_vllm.sh`.

```bash
# From repo root
python examples/generate_dialogue_vllm.py <model_owner> <model_name> <tensor_parallel_size> [--langs LANG ...] [--max-turns N] [--dataset NAME]
```

Examples:

```bash
# English only, 2 turn pairs (faster for testing)
python examples/generate_dialogue_vllm.py meta-llama Llama-3.3-70B-Instruct 4 --langs english --max-turns 2

# All default languages, 4 turn pairs (like the shell script)
python examples/generate_dialogue_vllm.py meta-llama Llama-3.3-70B-Instruct 4
```

Options:

- `--langs`: Languages to generate (default: chinese english french german portuguese spanish)
- `--max-turns`: Number of user/assistant turn pairs after turn 0 (default: 4)
- `--dataset`: Original dataset name without extension (default: ATOMIC10X_persona_1k_3)
- `--config`: Path to `config.yaml` (default: config.yaml)
- `--verbose`, `-v`: Verbose logging

Output layout matches the shell scripts: `batches_to_process/`, `completed_batches/`, and `dialogues/` under the configured paths.
