"""
Append batch completion results to a dialogue dataset (create new or update existing).
"""
from pathlib import Path
from typing import List, Optional

import orjson
from datasets import Dataset, load_from_disk
from tqdm import tqdm

from medal.config import Config
from medal.logging_config import get_logger
from medal.utils import load_jsonl, get_batch_paths

logger = get_logger(__name__)


def process_dialogue_batch(
    input_batch: str,
    role: str,
    lang: str,
    model: str,
    dialogue_file: Optional[str] = None,
    source_file: Optional[str] = None,
    config: Optional[Config] = None,
) -> str:
    """
    Merge batch completion JSONL into a dialogue dataset (create or update).

    Args:
        input_batch: Path to completion JSONL (each line: custom_id, response.body.choices[0].message.content).
        role: 'user' or 'assistant'.
        lang: Language code.
        model: Model name used for this batch.
        dialogue_file: Existing dialogue dataset path; if None, source_file must be provided to create new.
        source_file: JSONL of source requests (for new dataset); one line per dialogue.
        config: MEDAL config for paths.

    Returns:
        Path to the saved dialogue dataset.
    """
    if dialogue_file is None and source_file is None:
        raise ValueError("At least one of dialogue_file and source_file is required")
    cfg = config or Config()
    paths = get_batch_paths(
        input_batch,
        cfg.paths.dialogues,
        output_type="completed",
    )
    dial_path = Path(paths["output_dir"]) / paths["input_name"]
    dial_path.parent.mkdir(parents=True, exist_ok=True)
    batch_data = load_jsonl(input_batch)
    n = len(batch_data)
    if dialogue_file and Path(dialogue_file).exists():
        dataset = load_from_disk(dialogue_file)
        dialogue = list(dataset["dialogue"])
        models_list = list(dataset["models"])
        lang_list = list(dataset["lang"])
        source_list = list(dataset["source"])
        scene_list = list(dataset["scene"])
        ended_list = list(dataset["ended"])
        new = False
    else:
        dialogue = [[] for _ in range(n)]
        models_list = [[] for _ in range(n)]
        lang_list = [""] * n
        source_list = [""] * n
        scene_list = [""] * n
        ended_list = [False] * n
        new = True
        if source_file is None:
            raise ValueError("source_file required when creating new dialogue dataset")
        with open(source_file, "r", encoding="utf-8") as f:
            source_lines = f.readlines()
        if len([L for L in source_lines if L.strip()]) != n:
            raise ValueError("source_file line count != batch count")
    ended_count = 0
    for i, data_input in enumerate(tqdm(batch_data)):
        content = data_input["response"]["body"]["choices"][0]["message"]["content"]
        idx = int(data_input["custom_id"].split("-")[-1])
        if "END_OF_DIALOGUE" in content:
            ended_count += 1
            ended_list[idx] = True
        else:
            text = content.strip('"').strip("user: ")
            dialogue[idx].append({"role": role, "content": text})
            models_list[idx].append(model)
        if new:
            line = source_lines[idx].strip()
            src = orjson.loads(line.encode("utf-8")) if isinstance(line, str) else orjson.loads(line)
            lang_list[idx] = lang
            source_list[idx] = src["custom_id"]
            scene_list[idx] = src["body"]["messages"][-1]["content"]
            ended_list[idx] = False
    dataset = Dataset.from_dict({
        "source": source_list,
        "scene": scene_list,
        "lang": lang_list,
        "dialogue": dialogue,
        "models": models_list,
        "ended": ended_list,
    })
    dataset.save_to_disk(str(dial_path))
    logger.info("Saved dialogue dataset to %s", dial_path)
    logger.info("Dialogues ended: %.1f%%", 100.0 * sum(ended_list) / len(dataset))
    logger.info("New turns ended: %.1f%%", 100.0 * ended_count / len(dataset))
    return str(dial_path)
