"""
Narrative (starter) generation module for MEDAL framework.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional

from medal.config import Config, ModelConfig
from medal.logging_config import get_logger
from medal.prompts import NARRATE_SYS, EVALUATE_SYS
from medal.utils import (
    load_jsonl,
    save_jsonl,
    load_json,
    build_batch_path,
)

logger = get_logger(__name__)


class NarrativeGenerator:
    """
    Generator for narrative starters (turn 0 user messages) with validation and regeneration.
    """

    def __init__(
        self,
        dataset: str,
        lang: str,
        model: str,
        run_id: str = "vanilla",
        config: Optional[Config] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        self.dataset_path = dataset
        self.lang = lang
        self.model = model
        self.run_id = run_id
        self.config = config or Config()
        self.model_config = model_config or self.config.model

        self.dataset_name = Path(dataset).stem
        self.data: List[Dict[str, Any]] = load_json(dataset)

        base_path = build_batch_path(
            lang=lang,
            run_id=run_id,
            model_name=model,
            dataset_name=self.dataset_name,
            turn=0,
            file_type="gen",
            base_dir=self.config.paths.batches_to_process,
        )
        self.gen_file_path = str(base_path)
        self.eval_file_path = str(base_path).replace(".jsonl", "_eval.jsonl")
        self.regen_file_path = str(base_path).replace(".jsonl", "_regen.jsonl")

    def run(self) -> int:
        """Dispatch to generate, evaluate, or process (regenerate). Returns exit-like count for process."""
        raise NotImplementedError("Use generate(), evaluate(), or regenerate() directly")

    def generate(self) -> None:
        """Write generation requests for each scene to gen_file_path."""
        requests: List[Dict[str, Any]] = []
        model_base = self.model.split("/")[-1] if "/" in self.model else self.model
        for current_idx, scene in enumerate(self.data):
            idx = f"{self.run_id}-{self.dataset_name}-{self.lang}-{model_base}-{current_idx}"
            requests.append(self._build_request(idx, scene))
        save_jsonl(requests, self.gen_file_path)
        logger.info("Generated %d narrative requests to %s", len(requests), self.gen_file_path)

    def _completed_path(self, batches_path: str) -> Path:
        """Convert a path under batches_to_process to completed_batches."""
        base = self.config.paths.batches_to_process
        if batches_path.startswith(base + "/") or batches_path.startswith(base + "\\"):
            rel = batches_path[len(base) + 1:]
        else:
            rel = Path(batches_path).name
        return Path(self.config.paths.completed_batches) / rel

    def evaluate(self) -> None:
        """Write evaluation requests; uses regen output if present, else gen output."""
        regen_full = self._completed_path(self.regen_file_path)
        gen_full = self._completed_path(self.gen_file_path)
        if regen_full.exists():
            data_to_evaluate = load_jsonl(str(regen_full))
            logger.info("Evaluating regenerated narratives from %s", regen_full)
        else:
            data_to_evaluate = load_jsonl(str(gen_full))
        eval_requests = [self._build_eval_request(item) for item in data_to_evaluate]
        save_jsonl(eval_requests, self.eval_file_path)
        logger.info("Evaluation requests saved to %s", self.eval_file_path)

    def regenerate(self) -> int:
        """Identify failed evaluations, write regen requests, update completed gen file. Returns count to regen."""
        completed = Path(self.config.paths.completed_batches)
        gen_full = self._completed_path(self.gen_file_path)
        eval_full = self._completed_path(self.eval_file_path)
        rel_regen = self._completed_path(self.regen_file_path)

        gen_data = load_jsonl(str(gen_full))
        prior_regen_data: Optional[List[Dict[str, Any]]] = None
        if rel_regen.exists():
            prior_regen_data = load_jsonl(str(rel_regen))
        eval_data = load_jsonl(str(eval_full))

        regens_needed = 0
        edits = 0
        regen_requests: List[Dict[str, Any]] = []

        for current_idx, data_input in enumerate(eval_data):
            true_idx = int(data_input["custom_id"].split("-")[-1])
            content = data_input["response"]["body"]["choices"][0]["message"]["content"]
            if "No" in content:
                regens_needed += 1
                idx = data_input["custom_id"]
                scene = self.data[true_idx]
                regen_requests.append(self._build_request(idx, scene))
            else:
                if prior_regen_data is not None:
                    gen_data[true_idx] = prior_regen_data[current_idx]
                    edits += 1

        if regen_requests:
            save_jsonl(regen_requests, self.regen_file_path)
        if edits > 0:
            save_jsonl(gen_data, str(gen_full))
        logger.info("Examples that will be regenerated: %d.", regens_needed)
        return regens_needed

    def _build_request(self, idx: str, scene: Dict[str, Any]) -> Dict[str, Any]:
        """Build a single narrative generation request."""
        return {
            "custom_id": idx,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": NARRATE_SYS},
                    {"role": "user", "content": f"{scene}\nLanguage/Culture: {self.lang}"},
                ],
                "temperature": self.model_config.temperature,
                "top_p": self.model_config.top_p,
                "frequency_penalty": self.model_config.frequency_penalty,
                "presence_penalty": self.model_config.presence_penalty,
                "max_tokens": self.model_config.max_tokens,
            },
        }

    def _build_eval_request(self, data_input: Dict[str, Any]) -> Dict[str, Any]:
        """Build a single evaluation request for a narrative response."""
        return {
            "custom_id": data_input["custom_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gemini-2.0-flash",
                "messages": [
                    {"role": "system", "content": EVALUATE_SYS},
                    {
                        "role": "user",
                        "content": data_input["response"]["body"]["choices"][0]["message"]["content"],
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 64,
            },
        }
