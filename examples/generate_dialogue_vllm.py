#!/usr/bin/env python3
"""
Generate full multilingual dialogues using VLLM (mirrors generate_dialogues_vllm.sh).

Usage:
    python examples/generate_dialogue_vllm.py <model_owner> <model_name> <tensor_parallel_size> [--langs LANG ...] [--max-turns N] [--dataset NAME]

Example:
    python examples/generate_dialogue_vllm.py meta-llama Llama-3.3-70B-Instruct 4 --langs english --max-turns 2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medal.config import Config
from medal.logging_config import setup_logging, get_logger
from medal.tasks.narrative_generation import NarrativeGenerator
from medal.tasks.dialogue_generation import DialogueGenerator
from medal.agents import BatchAPIClient, run_vllm_batch
from medal.dialogues import process_dialogue_batch
from medal.utils import build_batch_path, build_dialogue_batch_path, build_dialogue_path

logger = get_logger(__name__)

DEFAULT_LANGS = ["chinese", "english", "french", "german", "portuguese", "spanish"]
USER_OWNER = "google"
USER_MODEL = "gemma-3-27b-it"
RUN_ID = "affective_persona"
MAX_STARTER_ITERATIONS = 10
MAX_USER_ITERATIONS = 5


def _run_starters(
    lang: str,
    original_dataset: str,
    tensor_parallel_size: int,
    config: Config,
    api_key_google: str | None,
) -> str:
    """Generate and validate turn-0 user starters; return path to dialogue dataset (to use as context for turn 0 assistant)."""
    dataset_path = f"tasks/narrative_generation/data/{original_dataset}.json"
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    full_user = f"{USER_OWNER}/{USER_MODEL}"
    nar = NarrativeGenerator(
        dataset=dataset_path,
        lang=lang,
        model=full_user,
        run_id=RUN_ID,
        config=config,
    )
    nar.generate()
    gen_path = str(build_batch_path(
        lang=lang,
        run_id=RUN_ID,
        model_name=full_user,
        dataset_name=Path(dataset_path).stem,
        turn=0,
        file_type="gen",
        base_dir=config.paths.batches_to_process,
    ))
    run_vllm_batch(gen_path, config=config, vllm_config=config.vllm)
    logger.info("Retrieved starter generations for %s", lang)
    nar.evaluate()
    eval_path = str(build_batch_path(
        lang=lang,
        run_id=RUN_ID,
        model_name=full_user,
        dataset_name=Path(dataset_path).stem,
        turn=0,
        file_type="eval",
        base_dir=config.paths.batches_to_process,
    ))
    if api_key_google:
        client = BatchAPIClient(provider="google", api_key=api_key_google, config=config)
        client.upload(eval_path, use_openai_batch=False)
    nar.regenerate()
    regens = nar.regenerate()
    iteration = 0
    while regens > 0 and iteration < MAX_STARTER_ITERATIONS:
        regen_path = str(build_batch_path(
            lang=lang,
            run_id=RUN_ID,
            model_name=full_user,
            dataset_name=Path(dataset_path).stem,
            turn=0,
            file_type="regen",
            base_dir=config.paths.batches_to_process,
        ))
        run_vllm_batch(regen_path, config=config, vllm_config=config.vllm)
        nar.evaluate()
        if api_key_google:
            client = BatchAPIClient(provider="google", api_key=api_key_google, config=config)
            client.upload(eval_path, use_openai_batch=False)
        regens = nar.regenerate()
        iteration += 1
        logger.info("Starter regens remaining: %s", regens)
    if iteration >= MAX_STARTER_ITERATIONS:
        logger.warning("Reached max starter iterations without completion")
    try:
        rel = Path(gen_path).relative_to(config.paths.batches_to_process)
    except ValueError:
        rel = Path(gen_path).name
    completed_gen = str(Path(config.paths.completed_batches) / rel)
    source_gen = gen_path
    dial_path = process_dialogue_batch(
        input_batch=completed_gen,
        role="user",
        lang=lang,
        model=USER_MODEL,
        source_file=source_gen,
        config=config,
    )
    return dial_path


def _run_turn0_assistant(
    lang: str,
    model_owner: str,
    model_name: str,
    original_dataset: str,
    tensor_parallel_size: int,
    context_dialogue_path: str,
    config: Config,
) -> None:
    """Generate turn-0 assistant responses and append to dialogue dataset."""
    full_model = f"{model_owner}/{model_name}"
    run_id = f"{USER_MODEL}_{model_name}"
    gen = DialogueGenerator(
        context=context_dialogue_path,
        lang=lang,
        model=full_model,
        role="assistant",
        turn=0,
        run_id=run_id,
        config=config,
    )
    gen.generate()
    gen_path = gen.gen_file_path
    run_vllm_batch(gen_path, config=config, vllm_config=config.vllm)
    completed_path = str(Path(config.paths.completed_batches) / Path(gen_path).relative_to(config.paths.batches_to_process))
    process_dialogue_batch(
        input_batch=completed_path,
        role="assistant",
        lang=lang,
        model=model_name,
        dialogue_file=context_dialogue_path,
        config=config,
    )


def _run_turn_x(
    lang: str,
    turn: int,
    model_owner: str,
    model_name: str,
    api_key_google: str | None,
    config: Config,
) -> None:
    """Generate one user turn (with validation) and one assistant turn."""
    full_model = f"{model_owner}/{model_name}"
    full_user = f"{USER_OWNER}/{USER_MODEL}"
    run_id_user = f"{model_name}_{USER_MODEL}"
    run_id_assistant = f"{USER_MODEL}_{model_name}"
    context_turn = turn - 1
    context_user = str(build_dialogue_path(lang=lang, run_id=run_id_user, model_name=full_user, turn=context_turn, base_dir=config.paths.dialogues))
    context_assistant = str(build_dialogue_path(lang=lang, run_id=run_id_assistant, model_name=full_model, turn=turn, base_dir=config.paths.dialogues))
    gen_user = DialogueGenerator(
        context=context_user,
        lang=lang,
        model=full_user,
        role="user",
        turn=turn,
        run_id=run_id_user,
        config=config,
    )
    gen_user.generate()
    out_dir_user = gen_user.gen_file_path
    run_vllm_batch(out_dir_user, config=config, vllm_config=config.vllm)
    completed_user = str(Path(config.paths.completed_batches) / Path(out_dir_user).relative_to(config.paths.batches_to_process))
    if api_key_google:
        gen_user.evaluate()
        eval_path = gen_user.eval_file_path
        client = BatchAPIClient(provider="google", api_key=api_key_google, config=config)
        client.upload(eval_path, use_openai_batch=False)
    regens = gen_user.regenerate()
    iteration = 0
    while regens > 0 and iteration < MAX_USER_ITERATIONS:
        run_vllm_batch(gen_user.regen_file_path, config=config, vllm_config=config.vllm)
        gen_user.evaluate()
        if api_key_google:
            client = BatchAPIClient(provider="google", api_key=api_key_google, config=config)
            client.upload(gen_user.eval_file_path, use_openai_batch=False)
        regens = gen_user.regenerate()
        iteration += 1
    process_dialogue_batch(
        input_batch=completed_user,
        role="user",
        lang=lang,
        model=USER_MODEL,
        dialogue_file=context_user,
        config=config,
    )
    gen_assistant = DialogueGenerator(
        context=context_assistant,
        lang=lang,
        model=full_model,
        role="assistant",
        turn=turn,
        run_id=run_id_assistant,
        config=config,
    )
    gen_assistant.generate()
    out_dir_assistant = gen_assistant.gen_file_path
    run_vllm_batch(out_dir_assistant, config=config, vllm_config=config.vllm)
    completed_assistant = str(Path(config.paths.completed_batches) / Path(out_dir_assistant).relative_to(config.paths.batches_to_process))
    process_dialogue_batch(
        input_batch=completed_assistant,
        role="assistant",
        lang=lang,
        model=model_name,
        dialogue_file=context_assistant,
        config=config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full dialogues with VLLM (mirrors generate_dialogues_vllm.sh)")
    parser.add_argument("model_owner", help="e.g. meta-llama")
    parser.add_argument("model_name", help="e.g. Llama-3.3-70B-Instruct")
    parser.add_argument("tensor_parallel_size", type=int, help="VLLM tensor parallel size")
    parser.add_argument("--langs", nargs="+", default=DEFAULT_LANGS, help="Languages to generate")
    parser.add_argument("--max-turns", type=int, default=4, help="Number of user/assistant turn pairs after turn 0")
    parser.add_argument("--dataset", default="ATOMIC10X_persona_1k_3", help="Original dataset name (no extension)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    setup_logging(level=10 if args.verbose else 20)
    config = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    config.vllm.tensor_parallel_size = args.tensor_parallel_size
    api_key_google = os.getenv("GEMENI_KEY") or os.getenv("GEMINI_KEY")
    for lang in args.langs:
        logger.info("=== Language: %s ===", lang)
        context_path = _run_starters(lang, args.dataset, args.tensor_parallel_size, config, api_key_google)
        _run_turn0_assistant(
            lang=lang,
            model_owner=args.model_owner,
            model_name=args.model_name,
            original_dataset=args.dataset,
            tensor_parallel_size=args.tensor_parallel_size,
            context_dialogue_path=context_path,
            config=config,
        )
        for turn in range(1, args.max_turns + 1):
            logger.info("=== %s turn %s ===", lang, turn)
            _run_turn_x(lang, turn, args.model_owner, args.model_name, api_key_google, config)
    logger.info("Done.")


if __name__ == "__main__":
    main()
