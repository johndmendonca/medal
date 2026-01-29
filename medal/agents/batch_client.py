"""
Batch API client: upload/download OpenAI-style batch jobs or run requests via async processor.
"""
import asyncio
import os
from pathlib import Path
from typing import Optional

import orjson
import openai

from medal.config import Config, APIConfig
from medal.logging_config import get_logger
from medal.utils import get_batch_paths, load_jsonl, save_jsonl

logger = get_logger(__name__)

def _process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
) -> None:
    # Use original implementation when available (run from project root)
    try:
        from agents.process_api_requests_from_file import process_api_requests_from_file as _run
    except ImportError:
        from medal.agents.api_processor import process_api_requests_from_file as _run
    asyncio.run(
        _run(
            requests_filepath=requests_filepath,
            save_filepath=save_filepath,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name=token_encoding_name,
            max_attempts=max_attempts,
            logging_level=logging_level,
        )
    )


PROVIDER_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    "azure": "https://gptreasoners.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview",
}


class BatchAPIClient:
    """
    Upload batch files to OpenAI (or similar) or process them via async API calls.
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.provider = provider
        self.config = config or Config()
        self.api_config = self.config.api
        if api_key is not None:
            self.api_config = APIConfig(
                provider=provider,
                api_key=api_key,
                max_requests_per_minute=self.api_config.max_requests_per_minute,
                max_tokens_per_minute=self.api_config.max_tokens_per_minute,
                max_attempts=self.api_config.max_attempts,
                token_encoding_name=self.api_config.token_encoding_name,
            )
        self._client: Optional[openai.OpenAI] = None
        self._url = PROVIDER_URLS.get(provider)
        if self._url is None:
            raise NotImplementedError(f"provider {provider} not implemented")

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            if self.provider != "openai":
                raise NotImplementedError(
                    "Batch upload/download is only supported for provider=openai; use async processing for other providers."
                )
            self._client = openai.OpenAI(api_key=self.api_config.api_key)
        return self._client

    def upload(
        self,
        input_file: str,
        use_openai_batch: bool = False,
    ) -> None:
        """
        Upload a batch file: either create an OpenAI batch job or process via async API.
        """
        if not Path(input_file).exists():
            raise FileNotFoundError(f"input file not found: {input_file}")
        paths = get_batch_paths(
            input_file,
            self.config.paths.completed_batches,
            output_type="completed",
        )
        if use_openai_batch and self.provider == "openai":
            self._upload_openai_batch(input_file, paths)
        else:
            self._process_async(input_file, paths)

    def _upload_openai_batch(self, input_file: str, paths: dict) -> None:
        client = self._get_client()
        comp = paths["components"]
        submitted_dir = (
            Path(self.config.paths.submitted_batches)
            / comp["three_up"]
            / comp["two_up"]
            / comp["one_up"]
        )
        submitted_dir.mkdir(parents=True, exist_ok=True)
        with open(input_file, "rb") as f:
            batch_file = client.files.create(file=f, purpose="batch")
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_job = client.batches.retrieve(batch_job.id)
        logger.info("Batch job for %s created with id %s", input_file, batch_job.id)
        meta = {
            "id": batch_job.id,
            "input_file_id_openai": batch_job.input_file_id,
            "input_file_local": input_file,
            "created_at": str(batch_job.created_at),
        }
        meta_path = submitted_dir / f"{paths['input_name']}.json"
        with open(meta_path, "w") as f:
            f.write(orjson.dumps(meta).decode("utf-8"))

    def _process_async(self, input_file: str, paths: dict) -> None:
        out_path = Path(paths["output_dir"]) / f"{paths['input_name']}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            logger.warning("Output file already exists, removing: %s", out_path)
            out_path.unlink()
        _process_api_requests_from_file(
            requests_filepath=input_file,
            save_filepath=str(out_path),
            request_url=self._url,
            api_key=self.api_config.api_key or "",
            max_requests_per_minute=float(self.api_config.max_requests_per_minute * 0.5),
            max_tokens_per_minute=float(self.api_config.max_tokens_per_minute * 0.5),
            token_encoding_name=self.api_config.token_encoding_name,
            max_attempts=self.api_config.max_attempts,
            logging_level=40,  # ERROR
        )
        # Sort and normalize format
        lines = load_jsonl(str(out_path))
        lines.sort(key=lambda x: x[0])
        normalized = [
            {"custom_id": line[1]["custom_id"], "response": {"body": line[2]}}
            for line in lines
        ]
        save_jsonl(normalized, str(out_path))
        logger.info("Saved processed batch to %s", out_path)

    def download(self, batch_id_file: str) -> None:
        """Download a completed OpenAI batch job result (OpenAI provider only)."""
        if self.provider != "openai":
            raise NotImplementedError("Download is only supported for provider=openai")
        data = orjson.loads(Path(batch_id_file).read_bytes())
        batch_id = data["id"]
        input_file = data["input_file_local"]
        paths = get_batch_paths(
            input_file,
            self.config.paths.completed_batches,
            output_type="completed",
        )
        client = self._get_client()
        batch_job = client.batches.retrieve(batch_id)
        logger.info("Batch job status: %s", batch_job.status)
        if batch_job.status != "completed":
            logger.warning("Batch job is not completed yet")
            return
        result_content = client.files.content(batch_job.output_file_id).content
        out_path = Path(paths["output_dir"]) / f"{paths['input_name']}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(result_content)
        logger.info("Downloaded results to %s", out_path)
