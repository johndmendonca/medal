# MEDAL: A Framework for Benchmarking LLMs as Multilingual Open-Domain Chatbots and Dialogue Evaluators

MEDAL is a framework for generating and evaluating multilingual open-domain chatbots and their evaluators. This framework supports various language models and provides a structured approach to creating conversational datasets.

## Paper

For a detailed understanding of the framework, methodology, and results, please refer to our paper:

[MEDAL: A Framework for Benchmarking LLMs as Multilingual Open-Domain Chatbots and Dialogue Evaluators](https://arxiv.org/abs/2505.22777)

```bibtex
@misc{mendonça2025medalframeworkbenchmarkingllms,
      title={MEDAL: A Framework for Benchmarking LLMs as Multilingual Open-Domain Chatbots and Dialogue Evaluators}, 
      author={John Mendonça and Alon Lavie and Isabel Trancoso},
      year={2025},
      eprint={2505.22777},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.22777}, 
}
```

## Data

The dataset generated by this framework, including GPT-4.1 annotations, can be downloaded from the [Hugging Face hub](https://huggingface.co/datasets/Johndfm/medal):
```python
from datasets import load_dataset

dataset = load_dataset("Johndfm/medal")
```

## Setup

### Prerequisites

*   Access to LLM APIs (OpenAI, Google Gemini, OpenRouter, etc.) or local LLM serving (e.g., VLLM)

### Environment Variables

The scripts in this framework rely on environment variables for API keys. Please set them in your environment:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export GEMENI_KEY="your_google_gemini_api_key" # Used for evaluation steps
export OPENROUTER_KEY="your_openrouter_api_key" # Used for some evaluation scripts
# Add any other API keys if you are using different providers
```

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/johndmendonca/medal.git
    cd medal
    ```

2.  Install the required Python packages. We recommend a conda environment:
    ```bash
    conda env create -f environment.yml
    ```

## Running the Framework

The framework is primarily controlled via shell scripts.

### Generating Dialogues

Dialogues are generated in turns, starting with user-generated "starters" (Turn 0), followed by alternating assistant and user turns. The framework supports generation using local models via VLLM or through various LLM provider APIs (like OpenAI) in batched or online mode.

**Key Scripts:**

*   `0_generate_starters_vllm.sh` / `0_generate_starters_openai.sh`: Generate and validate initial user messages.
*   `1_generate_turn0_vllm.sh`: Generate the first assistant response.
*   `2_upload_turnX_openAI.sh`: For OpenAI API, uploads user generation requests for a given turn.
*   `3_download_turnX_openAI.sh`: For OpenAI API, downloads completed user generations and runs validation.
*   `4_generate_turnX.sh`: Generates either a user turn (with validation) or an assistant turn (without validation) for subsequent turns. 

This script is used by both VLLM and OpenAI workflows.

**Orchestration Scripts:**

*   `generate_dialogues_vllm.sh`: Orchestrates the entire dialogue generation process for multiple languages and turns using VLLM-compatible models.
    ```bash
    # Example:
    ./generate_dialogues_vllm.sh <model_owner> <model_name> <tensor_parallel_size>
    ```

*   `generate_dialogues_openai.sh`: Orchestrates the dialogue generation process using OpenAI API for user turns and VLLM for assistant turns.
    ```bash
    # Example:
    ./generate_dialogues_openai.sh <user_model_openai> <assistant_model_owner_vllm> <assistant_model_name_vllm> <tensor_parallel_size_vllm>
    ```

**Workflow Overview:**

1.  **Turn 0 (User Starter Generation & Validation):**
    *   `tasks/narrative_generation/generate_narratives.py` is called to generate initial user messages.
    *   These messages are evaluated (e.g., using Gemini).
    *   Invalid messages are regenerated until criteria are met or max iterations are reached.
    *   `dialogues/process_batch.py` processes the validated starters into a dialogue dataset format.

2.  **Turn 0 (Assistant Response Generation):**
    *   `tasks/dialogue_generation/generate_turn.py` generates assistant responses to the user starters.
    *   `dialogues/process_batch.py` appends these responses to the dialogue dataset.

3.  **Subsequent Turns (User & Assistant):**
    *   **User Turns:**
        *   `tasks/dialogue_generation/generate_turn.py` generates user responses.
        *   These are evaluated (e.g., using Gemini).
        *   Invalid responses are regenerated.
        *   `dialogues/process_batch.py` appends validated user responses.
    *   **Assistant Turns:**
        *   `tasks/dialogue_generation/generate_turn.py` generates assistant responses.
        *   `dialogues/process_batch.py` appends these responses.

This loop continues for a predefined number of turns.

### Evaluating Dialogues

Once dialogues are generated, they can be evaluated for overall quality.

**Key Scripts:**

*   `tasks/dialogue_evaluation/evaluate_dialogue.py`: Prepares evaluation requests for a given set of dialogues.

**Orchestration Scripts:**

*   `evaluate_dialogues_mass.sh`: Evaluates a large set of dialogues.

```bash
# Example for evaluating a set of dialogues:
# (Ensure the 'dialogue' path in the script points to your generated dialogues)
./evaluate_dialogues_mass.sh
```
