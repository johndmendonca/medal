"""
Dialogue evaluation module for MEDAL framework.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional

from medal.config import Config, ModelConfig
from medal.logging_config import get_logger
from medal.prompts import SYS_PROMPT_HUMAN
from medal.utils import load_dataset, save_jsonl

logger = get_logger(__name__)


class DialogueEvaluator:
    """
    Prepares evaluation requests for complete dialogues (overall quality, dimensions).
    """

    def __init__(
        self,
        dialogue_path: str,
        lang: str,
        model: str,
        config: Optional[Config] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        self.dialogue_path = dialogue_path
        self.lang = lang
        self.model = model
        self.config = config or Config()
        self.model_config = model_config or self.config.model

        self.data = load_dataset(dialogue_path)
        model_base = model.split("/")[-1] if "/" in model else model
        self.out_dir = Path(self.config.paths.batches_to_process) / "evaluation" / Path(dialogue_path).name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.out_dir / f"{model_base}-{lang}.jsonl"

    def run(self) -> None:
        """Write evaluation requests to the output file."""
        requests: List[Dict[str, Any]] = []
        for current_idx, data_input in enumerate(self.data):
            dialogue = data_input.get("dialogue", [])
            dialogue_text = "\n".join(
                f"{turn['role']}: {turn['content']}" for turn in dialogue
            )
            message = [
                {"role": "system", "content": SYS_PROMPT_HUMAN},
                {"role": "user", "content": f"The Dialogue is as follows:\n{dialogue_text}"},
            ]
            custom_id = f"{Path(self.dialogue_path).name}-{current_idx}"
            call: Dict[str, Any] = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": message,
                    "temperature": self.model_config.temperature,
                    "top_p": self.model_config.top_p,
                    "max_tokens": self.model_config.max_tokens,
                    "response_format": {"type": "json_object"},
                },
            }
            requests.append(call)
        save_jsonl(requests, str(self.output_file))
        logger.info("Saved %d evaluation requests to %s", len(requests), self.output_file)
