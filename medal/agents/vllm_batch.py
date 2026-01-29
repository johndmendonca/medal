"""
VLLM batch inference: run a JSONL of chat completion requests through a local VLLM model.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional

import orjson
from vllm import LLM, SamplingParams

from medal.config import Config, VLLMConfig
from medal.logging_config import get_logger
from medal.utils import get_batch_paths, extract_path_components

logger = get_logger(__name__)


def _add_type_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Wrap each message content in [{type: 'text', text: content}] for models that require it."""
    return [
        {"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]}
        for msg in messages
    ]


def run_vllm_batch(
    input_file: str,
    output_file: Optional[str] = None,
    config: Optional[Config] = None,
    vllm_config: Optional[VLLMConfig] = None,
) -> str:
    """
    Run batch inference on a JSONL file of chat completion requests using VLLM.

    Args:
        input_file: Path to JSONL file; each line is a request with custom_id, body (model, messages, temperature, etc.).
        output_file: If None, derived from input path under completed_batches.
        config: MEDAL config for paths.
        vllm_config: VLLM settings (tensor_parallel_size, etc.).

    Returns:
        Path to the output JSONL file.
    """
    if not Path(input_file).exists():
        raise FileNotFoundError(f"input file not found: {input_file}")
    cfg = config or Config()
    vcfg = vllm_config or cfg.vllm
    if output_file is None:
        paths = get_batch_paths(
            input_file,
            cfg.paths.completed_batches,
            output_type="completed",
        )
        output_file = paths["output_file"]
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    custom_ids: List[str] = []
    prompts_to_process: List[str] = []
    llm: Optional[LLM] = None
    sampling_params = None
    model_name: Optional[str] = None
    job_body: Optional[Dict[str, Any]] = None
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            job = orjson.loads(line)
            custom_ids.append(job["custom_id"])
            body = job["body"]
            model_name = body["model"]
            if llm is None:
                llm = LLM(
                    model=model_name,
                    tensor_parallel_size=vcfg.tensor_parallel_size,
                    pipeline_parallel_size=vcfg.pipeline_parallel_size,
                    max_model_len=vcfg.max_model_len or 4096,
                    trust_remote_code=vcfg.trust_remote_code,
                    gpu_memory_utilization=vcfg.gpu_memory_utilization,
                )
                tokenizer = llm.get_tokenizer()
                sampling_params = SamplingParams(
                    temperature=body.get("temperature", 0.9),
                    max_tokens=body.get("max_tokens", 512),
                    top_p=body.get("top_p", 0.95),
                )
                job_body = body
            tokenizer = llm.get_tokenizer()
            messages = body["messages"]
            if "gemma-3" in model_name:
                messages = _add_type_messages(messages)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts_to_process.append(prompt)
    if llm is None or sampling_params is None or job_body is None:
        raise ValueError("No valid requests in input file")
    logger.info("Running VLLM batch with %d prompts", len(prompts_to_process))
    outputs = llm.generate(prompts_to_process, sampling_params)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, out in enumerate(outputs):
            obj = {
                "custom_id": custom_ids[i],
                "response": {
                    "body": {
                        "model": job_body["model"],
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": out.outputs[0].text,
                                },
                            },
                        ],
                    },
                },
                "prompt": out.prompt,
            }
            f.write(orjson.dumps(obj).decode("utf-8", "replace") + "\n")
    logger.info("Saved VLLM batch results to %s", output_file)
    return output_file
